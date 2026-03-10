#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "scheduling_env_core.h"
#include <vector>
#include <memory>

namespace py = pybind11;

// キャッシュのラッパークラス
class WindowCacheWrapper {
public:
    WindowCache* cache;
    bool owns_cache;  // キャッシュの所有権を持つかどうか
    
    WindowCacheWrapper(py::array_t<int32_t> window_status, int32_t H, int32_t W) {
        auto buf = window_status.request();
        const int32_t* data = static_cast<const int32_t*>(buf.ptr);
        cache = build_cache(data, H, W);
        if (!cache) {
            throw std::runtime_error("Failed to build cache");
        }
        owns_cache = true;
    }
    
    // WindowCache*から直接作成（所有権を移す）
    WindowCacheWrapper(WindowCache* c, bool own = true) : cache(c), owns_cache(own) {
    }
    
    ~WindowCacheWrapper() {
        if (cache && owns_cache) {
            free_cache(cache);
        }
    }
    
    // コピー禁止
    WindowCacheWrapper(const WindowCacheWrapper&) = delete;
    WindowCacheWrapper& operator=(const WindowCacheWrapper&) = delete;
    
    // ムーブ許可
    WindowCacheWrapper(WindowCacheWrapper&& other) noexcept : cache(other.cache), owns_cache(other.owns_cache) {
        other.cache = nullptr;
        other.owns_cache = false;
    }
};

// 割り当て結果のPythonタプルに変換
py::tuple allocation_result_to_tuple(const AllocationResult& result, int32_t job_width, int32_t job_height) {
    if (!result.found) {
        return py::make_tuple(py::none(), INFINITY);
    }
    
    if (!result.position.is_distributed) {
        // 連続割り当て
        int32_t i = result.position.pos.continuous.i;
        int32_t a = result.position.pos.continuous.a;
        return py::make_tuple(py::make_tuple(i, a), result.waiting_time);
    } else {
        // 分散割り当て
        int32_t a = result.position.pos.distributed.a;
        int32_t* node_allocation = result.position.pos.distributed.node_allocation;
        
        // ノード割り当てをPythonのリストに変換
        py::list node_allocation_list;
        int32_t idx = 0;
        for (int32_t col_offset = 0; col_offset < job_width; col_offset++) {
            py::list col_nodes;
            for (int32_t j = 0; j < job_height; j++) {
                col_nodes.append(node_allocation[idx++]);
            }
            node_allocation_list.append(col_nodes);
        }
        
        return py::make_tuple(py::make_tuple(0, a, node_allocation_list), result.waiting_time);
    }
}

// Pythonバインディング
PYBIND11_MODULE(scheduling_env_core, m) {
    m.doc() = "SchedulingEnv C言語実装のPythonバインディング";
    
    // WindowCacheWrapperクラス
    py::class_<WindowCacheWrapper>(m, "WindowCache")
        .def(py::init<py::array_t<int32_t>, int32_t, int32_t>(),
             "ウィンドウキャッシュを構築",
             py::arg("window_status"), py::arg("H"), py::arg("W"))
        .def(py::init([](py::array_t<int32_t> window_status, int32_t H, int32_t W, int32_t head) {
            auto buf = window_status.request();
            const int32_t* data = static_cast<const int32_t*>(buf.ptr);
            WindowCache* c = build_cache_from_ringbuffer(data, H, W, head);
            if (!c) throw std::runtime_error("Failed to build cache from ringbuffer");
            return new WindowCacheWrapper(c, true);
        }), "リングバッファからウィンドウキャッシュを構築",
             py::arg("window_status"), py::arg("H"), py::arg("W"), py::arg("head"));
    
    // find_allocation_position関数
    m.def("find_allocation_position",
          [](WindowCacheWrapper& cache_wrapper,
             int32_t job_width,
             int32_t job_height,
             int32_t when_submitted,
             int32_t current_time) {
              AllocationResult result = find_allocation_position(
                  cache_wrapper.cache,
                  job_width,
                  job_height,
                  when_submitted,
                  current_time
              );
              py::tuple ret = allocation_result_to_tuple(result, job_width, job_height);
              
              // 分散割り当ての場合はメモリを解放
              if (result.found && result.position.is_distributed && 
                  result.position.pos.distributed.node_allocation) {
                  free(result.position.pos.distributed.node_allocation);
              }
              
              return ret;
          },
          "割り当て位置を探索",
          py::arg("cache"),
          py::arg("job_width"),
          py::arg("job_height"),
          py::arg("when_submitted"),
          py::arg("current_time"));
    
    // time_transition_ringbuffer関数
    m.def("time_transition_ringbuffer",
          [](py::array_t<int32_t> window_status,
             py::array_t<int32_t> window_job_id,
             int32_t H,
             int32_t W,
             int32_t head) {
              if (!window_status.writeable() || !window_job_id.writeable()) {
                  throw std::runtime_error("Arrays must be writeable");
              }
              auto buf_status = window_status.request();
              auto buf_job_id = window_job_id.request();
              int32_t* status_ptr = static_cast<int32_t*>(buf_status.ptr);
              int32_t* job_id_ptr = static_cast<int32_t*>(buf_job_id.ptr);
              int32_t new_head = time_transition_ringbuffer(status_ptr, job_id_ptr, H, W, head);
              return py::make_tuple(window_status, window_job_id, new_head);
          },
          "リングバッファ版: 時間遷移",
          py::arg("window_status"), py::arg("window_job_id"),
          py::arg("H"), py::arg("W"), py::arg("head"));
    
    // do_schedule_ringbuffer関数
    m.def("do_schedule_ringbuffer",
          [](py::array_t<int32_t> window_status,
             py::array_t<int32_t> window_job_id,
             int32_t H,
             int32_t W,
             int32_t job_width,
             int32_t job_height,
             int32_t job_id,
             py::object position,
             int32_t head) {
              auto buf_status = window_status.request();
              auto buf_job_id = window_job_id.request();
              int32_t* status_ptr = static_cast<int32_t*>(buf_status.ptr);
              int32_t* job_id_ptr = static_cast<int32_t*>(buf_job_id.ptr);
              Position pos;
              py::tuple pos_tuple = py::cast<py::tuple>(position);
              if (py::len(pos_tuple) == 2) {
                  pos.is_distributed = false;
                  pos.pos.continuous.i = py::cast<int32_t>(pos_tuple[0]);
                  pos.pos.continuous.a = py::cast<int32_t>(pos_tuple[1]);
              } else if (py::len(pos_tuple) == 3) {
                  pos.is_distributed = true;
                  pos.pos.distributed.i = py::cast<int32_t>(pos_tuple[0]);
                  pos.pos.distributed.a = py::cast<int32_t>(pos_tuple[1]);
                  py::list node_allocation = py::cast<py::list>(pos_tuple[2]);
                  int32_t total_size = 0;
                  for (auto col_nodes : node_allocation) total_size += py::len(col_nodes);
                  pos.pos.distributed.node_allocation = (int32_t*)malloc(total_size * sizeof(int32_t));
                  pos.pos.distributed.allocation_size = total_size;
                  int32_t idx = 0;
                  for (auto col_nodes : node_allocation) {
                      py::list col_list = py::cast<py::list>(col_nodes);
                      for (auto node : col_list) {
                          pos.pos.distributed.node_allocation[idx++] = py::cast<int32_t>(node);
                      }
                  }
              } else {
                  throw std::runtime_error("Invalid position format");
              }
              do_schedule_ringbuffer(status_ptr, job_id_ptr, H, W,
                  job_width, job_height, job_id, &pos, head);
              if (pos.is_distributed && pos.pos.distributed.node_allocation) {
                  free(pos.pos.distributed.node_allocation);
              }
          },
          "リングバッファ版: ジョブスケジュール",
          py::arg("window_status"), py::arg("window_job_id"),
          py::arg("H"), py::arg("W"), py::arg("job_width"), py::arg("job_height"),
          py::arg("job_id"), py::arg("position"), py::arg("head"));
    
    // time_transition関数
    m.def("time_transition",
          [](py::array_t<int32_t> window_status,
             py::array_t<int32_t> window_job_id,
             int32_t H,
             int32_t W,
             bool slide) {
              // 書き込み可能であることを確認
              if (!window_status.writeable()) {
                  throw std::runtime_error("window_status is not writeable");
              }
              if (!window_job_id.writeable()) {
                  throw std::runtime_error("window_job_id is not writeable");
              }
              
              auto buf_status = window_status.request();
              auto buf_job_id = window_job_id.request();
              
              // C連続であることを確認
              if (buf_status.ndim != 2 || buf_job_id.ndim != 2) {
                  throw std::runtime_error("Arrays must be 2D");
              }
              
              int32_t* status_ptr = static_cast<int32_t*>(buf_status.ptr);
              int32_t* job_id_ptr = static_cast<int32_t*>(buf_job_id.ptr);
              
              // 配列を直接変更（in-place）
              time_transition(status_ptr, job_id_ptr, H, W, slide);
              
              // 配列を返す（既存のPython実装と互換性を保つ）
              return py::make_tuple(window_status, window_job_id);
          },
          "時間遷移（スライドウィンドウ）",
          py::arg("window_status"),
          py::arg("window_job_id"),
          py::arg("H"),
          py::arg("W"),
          py::arg("slide"));
    
    // do_schedule関数
    m.def("do_schedule",
          [](py::array_t<int32_t> window_status,
             py::array_t<int32_t> window_job_id,
             int32_t H,
             int32_t W,
             int32_t job_width,
             int32_t job_height,
             int32_t job_id,
             py::object position) {
              auto buf_status = window_status.request();
              auto buf_job_id = window_job_id.request();
              
              int32_t* status_ptr = static_cast<int32_t*>(buf_status.ptr);
              int32_t* job_id_ptr = static_cast<int32_t*>(buf_job_id.ptr);
              
              Position pos;
              
              // Pythonのタプルから位置情報を取得
              py::tuple pos_tuple = py::cast<py::tuple>(position);
              if (py::len(pos_tuple) == 2) {
                  // 連続割り当て
                  pos.is_distributed = false;
                  pos.pos.continuous.i = py::cast<int32_t>(pos_tuple[0]);
                  pos.pos.continuous.a = py::cast<int32_t>(pos_tuple[1]);
              } else if (py::len(pos_tuple) == 3) {
                  // 分散割り当て
                  pos.is_distributed = true;
                  pos.pos.distributed.i = py::cast<int32_t>(pos_tuple[0]);
                  pos.pos.distributed.a = py::cast<int32_t>(pos_tuple[1]);
                  py::list node_allocation = py::cast<py::list>(pos_tuple[2]);
                  
                  // ノード割り当てをフラット化
                  int32_t total_size = 0;
                  for (auto col_nodes : node_allocation) {
                      total_size += py::len(col_nodes);
                  }
                  
                  pos.pos.distributed.node_allocation = 
                      (int32_t*)malloc(total_size * sizeof(int32_t));
                  pos.pos.distributed.allocation_size = total_size;
                  
                  int32_t idx = 0;
                  for (auto col_nodes : node_allocation) {
                      py::list col_list = py::cast<py::list>(col_nodes);
                      for (auto node : col_list) {
                          pos.pos.distributed.node_allocation[idx++] = 
                              py::cast<int32_t>(node);
                      }
                  }
              } else {
                  throw std::runtime_error("Invalid position format");
              }
              
              do_schedule(status_ptr, job_id_ptr, H, W, 
                         job_width, job_height, job_id, &pos);
              
              // 分散割り当ての場合はメモリを解放
              if (pos.is_distributed && pos.pos.distributed.node_allocation) {
                  free(pos.pos.distributed.node_allocation);
              }
          },
          "ジョブのスケジュール実行",
          py::arg("window_status"),
          py::arg("window_job_id"),
          py::arg("H"),
          py::arg("W"),
          py::arg("job_width"),
          py::arg("job_height"),
          py::arg("job_id"),
          py::arg("position"));
    
    // get_unique_job_ids関数
    m.def("get_unique_job_ids",
          [](py::array_t<int32_t> history_matrix,
             int32_t H,
             int32_t W,
             int32_t max_job_id) {
              auto buf = history_matrix.request();
              const int32_t* data = static_cast<const int32_t*>(buf.ptr);
              
              int32_t count;
              int32_t* result = get_unique_job_ids(data, H, W, max_job_id, &count);
              
              if (!result) {
                  return py::array_t<int32_t>(0);
              }
              
              // NumPy配列に変換
              py::array_t<int32_t> py_result(count);
              auto buf_result = py_result.request();
              int32_t* result_ptr = static_cast<int32_t*>(buf_result.ptr);
              memcpy(result_ptr, result, count * sizeof(int32_t));
              
              free(result);
              return py_result;
          },
          "ユニークなジョブIDを取得",
          py::arg("history_matrix"),
          py::arg("H"),
          py::arg("W"),
          py::arg("max_job_id"));
    
    // calculate_makespan関数
    m.def("calculate_makespan",
          [](py::array_t<int32_t> window_matrix,
             int32_t H,
             int32_t W) {
              auto buf = window_matrix.request();
              const int32_t* data = static_cast<const int32_t*>(buf.ptr);
              
              return calculate_makespan(data, H, W);
          },
          "makespanを計算",
          py::arg("window_matrix"),
          py::arg("H"),
          py::arg("W"));
    
    // update_cache_incremental関数
    m.def("update_cache_incremental",
          [](WindowCacheWrapper& cache_wrapper,
             py::array_t<int32_t> window_status,
             int32_t i_start,
             int32_t i_end,
             int32_t a_start,
             int32_t a_end) {
              auto buf = window_status.request();
              const int32_t* data = static_cast<const int32_t*>(buf.ptr);
              
              update_cache_incremental(cache_wrapper.cache, data, 
                                      i_start, i_end, a_start, a_end);
          },
          "キャッシュの差分更新（ジョブ追加時）",
          py::arg("cache"),
          py::arg("window_status"),
          py::arg("i_start"),
          py::arg("i_end"),
          py::arg("a_start"),
          py::arg("a_end"));
    
    // update_cache_time_transition関数
    m.def("update_cache_time_transition",
          [](WindowCacheWrapper& cache_wrapper,
             py::array_t<int32_t> window_status) {
              auto buf = window_status.request();
              const int32_t* data = static_cast<const int32_t*>(buf.ptr);
              
              update_cache_time_transition(cache_wrapper.cache, data);
          },
          "キャッシュの差分更新（時間遷移時）",
          py::arg("cache"),
          py::arg("window_status"));

    // update_cache_time_transition_ringbuffer関数
    m.def("update_cache_time_transition_ringbuffer",
          [](WindowCacheWrapper& cache_wrapper) {
              update_cache_time_transition_ringbuffer(cache_wrapper.cache);
          },
          "リングバッファ版: キャッシュの差分更新（時間遷移時）",
          py::arg("cache"));

    // update_cache_incremental_ringbuffer関数
    m.def("update_cache_incremental_ringbuffer",
          [](WindowCacheWrapper& cache_wrapper,
             py::array_t<int32_t> window_status,
             int32_t i_start,
             int32_t i_end,
             int32_t a_start,
             int32_t a_end,
             int32_t head) {
              auto buf = window_status.request();
              const int32_t* data = static_cast<const int32_t*>(buf.ptr);
              update_cache_incremental_ringbuffer(cache_wrapper.cache, data,
                                                  i_start, i_end, a_start, a_end, head);
          },
          "リングバッファ版: キャッシュの差分更新（ジョブ追加時）",
          py::arg("cache"),
          py::arg("window_status"),
          py::arg("i_start"),
          py::arg("i_end"),
          py::arg("a_start"),
          py::arg("a_end"),
          py::arg("head"));
    
    // get_observation関数
    m.def("get_observation",
          [](py::array_t<int32_t> onpre_status,
             py::array_t<int32_t> cloud_status,
             py::array_t<double> job_queue,
             int32_t H_onpre,
             int32_t H_cloud,
             int32_t W,
             int32_t obs_window_size) {
              auto buf_onpre = onpre_status.request();
              auto buf_cloud = cloud_status.request();
              auto buf_job = job_queue.request();

              const int32_t* onpre_ptr = static_cast<const int32_t*>(buf_onpre.ptr);
              const int32_t* cloud_ptr = static_cast<const int32_t*>(buf_cloud.ptr);
              const double* job_ptr = static_cast<const double*>(buf_job.ptr);

              size_t out_size = H_onpre * obs_window_size + H_cloud * obs_window_size + 40;
              py::array_t<float> output(out_size);
              auto buf_out = output.request();
              float* out_ptr = static_cast<float*>(buf_out.ptr);

              get_observation(onpre_ptr, cloud_ptr, job_ptr,
                             H_onpre, H_cloud, W, obs_window_size, out_ptr);

              return output;
          },
          "観測データをC側で作成",
          py::arg("onpre_status"),
          py::arg("cloud_status"),
          py::arg("job_queue"),
          py::arg("H_onpre"),
          py::arg("H_cloud"),
          py::arg("W"),
          py::arg("obs_window_size"));

    // get_observation_ringbuffer関数（Python側の配列構築を省略）
    m.def("get_observation_ringbuffer",
          [](py::array_t<int32_t> onpre_status,
             py::array_t<int32_t> cloud_status,
             py::array_t<double> job_queue,
             int32_t H_onpre,
             int32_t H_cloud,
             int32_t W,
             int32_t head_onpre,
             int32_t head_cloud,
             int32_t obs_window_size) {
              auto buf_onpre = onpre_status.request();
              auto buf_cloud = cloud_status.request();
              auto buf_job = job_queue.request();
              const int32_t* onpre_ptr = static_cast<const int32_t*>(buf_onpre.ptr);
              const int32_t* cloud_ptr = static_cast<const int32_t*>(buf_cloud.ptr);
              const double* job_ptr = static_cast<const double*>(buf_job.ptr);
              size_t out_size = H_onpre * obs_window_size + H_cloud * obs_window_size + 40;
              py::array_t<float> output(out_size);
              float* out_ptr = static_cast<float*>(output.request().ptr);
              get_observation_ringbuffer(onpre_ptr, cloud_ptr, job_ptr,
                  H_onpre, H_cloud, W, head_onpre, head_cloud, obs_window_size, out_ptr);
              return output;
          },
          "リングバッファ版: 観測を直接構築",
          py::arg("onpre_status"),
          py::arg("cloud_status"),
          py::arg("job_queue"),
          py::arg("H_onpre"),
          py::arg("H_cloud"),
          py::arg("W"),
          py::arg("head_onpre"),
          py::arg("head_cloud"),
          py::arg("obs_window_size"));

    // rebuild_cache_if_needed関数
    m.def("rebuild_cache_if_needed",
          [](py::object cache_obj,  // WindowCacheまたはNone
             py::array_t<int32_t> window_status,
             int32_t H,
             int32_t W,
             int32_t current_version,
             int32_t cache_version,
             bool window_changed) {
              auto buf = window_status.request();
              const int32_t* data = static_cast<const int32_t*>(buf.ptr);
              
              WindowCache* cache = nullptr;
              
              if (!cache_obj.is_none()) {
                  // 既存のキャッシュを取得
                  try {
                      WindowCacheWrapper& wrapper = cache_obj.cast<WindowCacheWrapper&>();
                      cache = wrapper.cache;
                      // 所有権を移す（C関数が管理するため）
                      wrapper.owns_cache = false;
                  } catch (const py::cast_error&) {
                      // キャストに失敗した場合はNoneとして扱う
                      cache = nullptr;
                  }
              }
              
              int32_t cache_ver = cache_version;
              bool changed = window_changed;
              
              // C関数を呼び出し
              cache = rebuild_cache_if_needed(
                  cache, data, H, W, current_version, &cache_ver, &changed
              );
              
              // 結果を返す（キャッシュ、バージョン、変更フラグ）
              py::object result_cache;
              if (cache) {
                  // 新しいWindowCacheWrapperを作成（所有権を持つ）
                  WindowCacheWrapper* wrapper = new WindowCacheWrapper(cache, true);
                  result_cache = py::cast(wrapper);
              } else {
                  result_cache = py::none();
              }
              return py::make_tuple(result_cache, cache_ver, changed);
          },
          "キャッシュの再構築（最適化版）",
          py::arg("cache"),
          py::arg("window_status"),
          py::arg("H"),
          py::arg("W"),
          py::arg("current_version"),
          py::arg("cache_version"),
          py::arg("window_changed"));
}


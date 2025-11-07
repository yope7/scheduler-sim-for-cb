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
    
    WindowCacheWrapper(py::array_t<int32_t> window_status, int32_t H, int32_t W) {
        auto buf = window_status.request();
        const int32_t* data = static_cast<const int32_t*>(buf.ptr);
        cache = build_cache(data, H, W);
        if (!cache) {
            throw std::runtime_error("Failed to build cache");
        }
    }
    
    ~WindowCacheWrapper() {
        if (cache) {
            free_cache(cache);
        }
    }
    
    // コピー禁止
    WindowCacheWrapper(const WindowCacheWrapper&) = delete;
    WindowCacheWrapper& operator=(const WindowCacheWrapper&) = delete;
    
    // ムーブ許可
    WindowCacheWrapper(WindowCacheWrapper&& other) noexcept : cache(other.cache) {
        other.cache = nullptr;
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
             py::arg("window_status"), py::arg("H"), py::arg("W"));
    
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
}


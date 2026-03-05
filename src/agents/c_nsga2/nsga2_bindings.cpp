#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "nsga2_core.h"
#include <vector>
#include <memory>
#include <random>

namespace py = pybind11;

// Pythonバインディング
PYBIND11_MODULE(nsga2_core, m) {
    m.doc() = "NSGA-II C言語実装のPythonバインディング";
    
    // dominates関数
    m.def("dominates",
          [](py::array_t<double> objectives1, py::array_t<double> objectives2) {
              auto buf1 = objectives1.request();
              auto buf2 = objectives2.request();
              
              if (buf1.ndim != 1 || buf2.ndim != 1) {
                  throw std::runtime_error("Objectives must be 1D arrays");
              }
              if (buf1.size != buf2.size) {
                  throw std::runtime_error("Objectives must have the same size");
              }
              
              const double* obj1 = static_cast<const double*>(buf1.ptr);
              const double* obj2 = static_cast<const double*>(buf2.ptr);
              
              return dominates(obj1, obj2, buf1.size);
          },
          "支配関係を判定",
          py::arg("objectives1"), py::arg("objectives2"));
    
    // non_dominated_sort関数
    m.def("non_dominated_sort",
          [](py::array_t<double> objectives_matrix) {
              auto buf = objectives_matrix.request();
              
              if (buf.ndim != 2) {
                  throw std::runtime_error("Objectives matrix must be 2D");
              }
              
              int32_t n_pop = buf.shape[0];
              int32_t n_obj = buf.shape[1];
              
              const double* obj_matrix = static_cast<const double*>(buf.ptr);
              int32_t* ranks = (int32_t*)malloc(n_pop * sizeof(int32_t));
              
              non_dominated_sort(obj_matrix, ranks, n_pop, n_obj);
              
              // NumPy配列に変換
              py::array_t<int32_t> result = py::array_t<int32_t>(n_pop);
              auto result_buf = result.request();
              int32_t* result_ptr = static_cast<int32_t*>(result_buf.ptr);
              memcpy(result_ptr, ranks, n_pop * sizeof(int32_t));
              
              free(ranks);
              return result;
          },
          "非支配ソートを実行",
          py::arg("objectives_matrix"));
    
    // calculate_crowding_distance関数
    m.def("calculate_crowding_distance",
          [](py::array_t<double> objectives_matrix) {
              auto buf = objectives_matrix.request();
              
              if (buf.ndim != 2) {
                  throw std::runtime_error("Objectives matrix must be 2D");
              }
              
              int32_t n_pop = buf.shape[0];
              int32_t n_obj = buf.shape[1];
              
              const double* obj_matrix = static_cast<const double*>(buf.ptr);
              double* distances = (double*)malloc(n_pop * sizeof(double));
              
              calculate_crowding_distance(obj_matrix, distances, n_pop, n_obj);
              
              // NumPy配列に変換
              py::array_t<double> result = py::array_t<double>(n_pop);
              auto result_buf = result.request();
              double* result_ptr = static_cast<double*>(result_buf.ptr);
              memcpy(result_ptr, distances, n_pop * sizeof(double));
              
              free(distances);
              return result;
          },
          "混雑度を計算",
          py::arg("objectives_matrix"));
    
    // tournament_selection関数
    m.def("tournament_selection",
          [](py::array_t<double> objectives_matrix,
             py::array_t<int32_t> ranks,
             py::array_t<double> crowding_distances,
             int32_t selection_size,
             int32_t tournament_size,
             uint32_t random_seed) {
              auto buf_obj = objectives_matrix.request();
              auto buf_ranks = ranks.request();
              auto buf_dist = crowding_distances.request();
              
              if (buf_obj.ndim != 2) {
                  throw std::runtime_error("Objectives matrix must be 2D");
              }
              if (buf_ranks.ndim != 1 || buf_dist.ndim != 1) {
                  throw std::runtime_error("Ranks and distances must be 1D");
              }
              
              int32_t n_pop = buf_obj.shape[0];
              int32_t n_obj = buf_obj.shape[1];
              
              const double* obj_matrix = static_cast<const double*>(buf_obj.ptr);
              const int32_t* ranks_ptr = static_cast<const int32_t*>(buf_ranks.ptr);
              const double* dist_ptr = static_cast<const double*>(buf_dist.ptr);
              
              int32_t* selected = (int32_t*)malloc(selection_size * sizeof(int32_t));
              
              tournament_selection(obj_matrix, ranks_ptr, dist_ptr, selected,
                                  n_pop, n_obj, selection_size, tournament_size, random_seed);
              
              // NumPy配列に変換
              py::array_t<int32_t> result = py::array_t<int32_t>(selection_size);
              auto result_buf = result.request();
              int32_t* result_ptr = static_cast<int32_t*>(result_buf.ptr);
              memcpy(result_ptr, selected, selection_size * sizeof(int32_t));
              
              free(selected);
              return result;
          },
          "トーナメント選択を実行",
          py::arg("objectives_matrix"), py::arg("ranks"),
          py::arg("crowding_distances"), py::arg("selection_size"),
          py::arg("tournament_size"), py::arg("random_seed"));
    
    // single_point_crossover関数
    m.def("single_point_crossover",
          [](py::array_t<int32_t> parent1,
             py::array_t<int32_t> parent2,
             int32_t crossover_point) {
              auto buf1 = parent1.request();
              auto buf2 = parent2.request();
              
              if (buf1.ndim != 1 || buf2.ndim != 1) {
                  throw std::runtime_error("Parents must be 1D arrays");
              }
              if (buf1.size != buf2.size) {
                  throw std::runtime_error("Parents must have the same size");
              }
              
              int32_t length = buf1.size;
              const int32_t* p1 = static_cast<const int32_t*>(buf1.ptr);
              const int32_t* p2 = static_cast<const int32_t*>(buf2.ptr);
              
              int32_t* child1 = (int32_t*)malloc(length * sizeof(int32_t));
              int32_t* child2 = (int32_t*)malloc(length * sizeof(int32_t));
              
              single_point_crossover(p1, p2, child1, child2, length, crossover_point);
              
              py::array_t<int32_t> result1 = py::array_t<int32_t>(length);
              py::array_t<int32_t> result2 = py::array_t<int32_t>(length);
              auto buf_r1 = result1.request();
              auto buf_r2 = result2.request();
              memcpy(buf_r1.ptr, child1, length * sizeof(int32_t));
              memcpy(buf_r2.ptr, child2, length * sizeof(int32_t));
              
              free(child1);
              free(child2);
              return py::make_tuple(result1, result2);
          },
          "一点交叉を実行",
          py::arg("parent1"), py::arg("parent2"), py::arg("crossover_point"));
    
    // two_point_crossover関数
    m.def("two_point_crossover",
          [](py::array_t<int32_t> parent1,
             py::array_t<int32_t> parent2,
             int32_t point1,
             int32_t point2) {
              auto buf1 = parent1.request();
              auto buf2 = parent2.request();
              
              if (buf1.ndim != 1 || buf2.ndim != 1) {
                  throw std::runtime_error("Parents must be 1D arrays");
              }
              if (buf1.size != buf2.size) {
                  throw std::runtime_error("Parents must have the same size");
              }
              
              int32_t length = buf1.size;
              const int32_t* p1 = static_cast<const int32_t*>(buf1.ptr);
              const int32_t* p2 = static_cast<const int32_t*>(buf2.ptr);
              
              int32_t* child1 = (int32_t*)malloc(length * sizeof(int32_t));
              int32_t* child2 = (int32_t*)malloc(length * sizeof(int32_t));
              
              two_point_crossover(p1, p2, child1, child2, length, point1, point2);
              
              py::array_t<int32_t> result1 = py::array_t<int32_t>(length);
              py::array_t<int32_t> result2 = py::array_t<int32_t>(length);
              auto buf_r1 = result1.request();
              auto buf_r2 = result2.request();
              memcpy(buf_r1.ptr, child1, length * sizeof(int32_t));
              memcpy(buf_r2.ptr, child2, length * sizeof(int32_t));
              
              free(child1);
              free(child2);
              return py::make_tuple(result1, result2);
          },
          "二点交叉を実行",
          py::arg("parent1"), py::arg("parent2"), py::arg("point1"), py::arg("point2"));
    
    // uniform_crossover関数
    m.def("uniform_crossover",
          [](py::array_t<int32_t> parent1,
             py::array_t<int32_t> parent2,
             py::array_t<int32_t> mask) {
              auto buf1 = parent1.request();
              auto buf2 = parent2.request();
              auto buf_mask = mask.request();
              
              if (buf1.ndim != 1 || buf2.ndim != 1 || buf_mask.ndim != 1) {
                  throw std::runtime_error("All inputs must be 1D arrays");
              }
              if (buf1.size != buf2.size || buf1.size != buf_mask.size) {
                  throw std::runtime_error("All inputs must have the same size");
              }
              
              int32_t length = buf1.size;
              const int32_t* p1 = static_cast<const int32_t*>(buf1.ptr);
              const int32_t* p2 = static_cast<const int32_t*>(buf2.ptr);
              const int32_t* m = static_cast<const int32_t*>(buf_mask.ptr);
              
              int32_t* child1 = (int32_t*)malloc(length * sizeof(int32_t));
              int32_t* child2 = (int32_t*)malloc(length * sizeof(int32_t));
              
              uniform_crossover(p1, p2, child1, child2, m, length);
              
              py::array_t<int32_t> result1 = py::array_t<int32_t>(length);
              py::array_t<int32_t> result2 = py::array_t<int32_t>(length);
              auto buf_r1 = result1.request();
              auto buf_r2 = result2.request();
              memcpy(buf_r1.ptr, child1, length * sizeof(int32_t));
              memcpy(buf_r2.ptr, child2, length * sizeof(int32_t));
              
              free(child1);
              free(child2);
              return py::make_tuple(result1, result2);
          },
          "一様交叉を実行",
          py::arg("parent1"), py::arg("parent2"), py::arg("mask"));
    
    // mutation関数
    m.def("mutation",
          [](py::array_t<int32_t> chromosome,
             py::array_t<int32_t> mutation_mask) {
              auto buf_chr = chromosome.request();
              auto buf_mask = mutation_mask.request();
              
              if (buf_chr.ndim != 1 || buf_mask.ndim != 1) {
                  throw std::runtime_error("Inputs must be 1D arrays");
              }
              if (buf_chr.size != buf_mask.size) {
                  throw std::runtime_error("Inputs must have the same size");
              }
              
              int32_t length = buf_chr.size;
              // mutable_uncheckedを使用して配列を直接変更
              auto chr_buf = chromosome.mutable_unchecked<1>();
              const int32_t* mask = static_cast<const int32_t*>(buf_mask.ptr);
              
              // 配列を変更（in-place）
              int32_t* chr = chr_buf.mutable_data(0);
              mutation(chr, mask, length);
              
              return chromosome;  // 入力配列をそのまま返す（in-place変更）
          },
          "突然変異を実行",
          py::arg("chromosome"), py::arg("mutation_mask"));
    
    // eliminate_duplicates関数
    m.def("eliminate_duplicates",
          [](py::array_t<int32_t> chromosomes) {
              auto buf = chromosomes.request();
              
              if (buf.ndim != 2) {
                  throw std::runtime_error("Chromosomes must be 2D array");
              }
              
              int32_t n_pop = buf.shape[0];
              int32_t chromosome_length = buf.shape[1];
              
              const int32_t* chr = static_cast<const int32_t*>(buf.ptr);
              int32_t* unique_indices = (int32_t*)malloc(n_pop * sizeof(int32_t));
              int32_t n_unique = 0;
              
              eliminate_duplicates(chr, unique_indices, &n_unique, n_pop, chromosome_length);
              
              // NumPy配列に変換（ユニークな個体のみ）
              py::array_t<int32_t> result = py::array_t<int32_t>(n_unique);
              auto result_buf = result.request();
              int32_t* result_ptr = static_cast<int32_t*>(result_buf.ptr);
              memcpy(result_ptr, unique_indices, n_unique * sizeof(int32_t));
              
              free(unique_indices);
              return result;
          },
          "重複個体を排除",
          py::arg("chromosomes"));
    
    // 並列評価のスレッド初期化
    m.def("init_evaluation_threads",
          [](int32_t n_threads) {
              init_evaluation_threads(n_threads);
          },
          "並列評価のスレッド数を設定",
          py::arg("n_threads") = 0);
    
    // 並列評価のクリーンアップ
    m.def("cleanup_evaluation_threads",
          []() {
              cleanup_evaluation_threads();
          },
          "並列評価のクリーンアップ");
    
    // evaluate_individual_c関数（コールバック関数を使用）
    // 注意: コールバック関数はPythonから提供されるため、直接バインディングは難しい
    // 代わりに、Python側で評価ループを実装し、C実装版の非優劣ソートなどを使用する
    
    // evaluate_population_c関数（コールバック関数を使用）
    // 注意: 同上
}


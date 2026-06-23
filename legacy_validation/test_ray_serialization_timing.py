"""
Rayのシリアライゼーション時間を詳細に測定するスクリプト
"""
import numpy as np
import time
import ray
import yaml
from src.agents.pcn_agent import Transition
from src.envs.scheduling_variants.bitmap_c_env import SchedulingEnvCacheOptimized
from src.utils.job_gen.job_generator import JobGenerator
from src.agents.pcn_agent import PCN
import cProfile
import pstats
import io
from pstats import SortKey

# Rayの初期化
ray.init(ignore_reinit_error=True, object_store_memory=4 * 1024 * 1024 * 1024)

# 設定ファイルの読み込み
with open('config/config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

N_JOBS = 12
N_ACTORS = 16
EPISODES_PER_ACTOR = 1

# =========================
# ReplayBuffer (Ray Actor)
# =========================
@ray.remote
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
        self.episode_hashes = set()
        self._hash_cache = {}
        self.timing_stats = {
            'add_batch_calls': 0,
            'add_batch_time': 0.0,
            'get_all_episodes_calls': 0,
            'get_all_episodes_time': 0.0,
            'serialization_time': 0.0,
            'deserialization_time': 0.0,
        }
        print(f"ReplayBuffer initialized with max_size={max_size}")

    def add_batch(self, episodes):
        """エピソードをバッチで追加（タイミング測定付き）"""
        start_time = time.time()
        
        # シリアライゼーション時間を測定（Rayが自動的に行う）
        serialization_start = time.time()
        
        added_count = 0
        skipped_count = 0
        
        for episode in episodes:
            episode_hash = self._compute_episode_hash(episode)
            if episode_hash in self.episode_hashes:
                skipped_count += 1
                continue
            
            if len(self.buffer) >= self.max_size:
                oldest_episode = self.buffer.pop(0)
                oldest_hash = self._compute_episode_hash(oldest_episode)
                self.episode_hashes.discard(oldest_hash)
                oldest_episode_id = id(oldest_episode)
                self._hash_cache.pop(oldest_episode_id, None)
            
            self.buffer.append(episode)
            self.episode_hashes.add(episode_hash)
            added_count += 1
        
        serialization_end = time.time()
        end_time = time.time()
        
        self.timing_stats['add_batch_calls'] += 1
        self.timing_stats['add_batch_time'] += (end_time - start_time)
        self.timing_stats['serialization_time'] += (serialization_end - serialization_start)
        
        return added_count

    def _compute_episode_hash(self, episode):
        """エピソードのハッシュ値を計算"""
        import hashlib
        
        if not episode:
            return 0
        
        episode_id = id(episode)
        if episode_id in self._hash_cache:
            return self._hash_cache[episode_id]
        
        hasher = hashlib.md5()
        episode_len = len(episode)
        hasher.update(episode_len.to_bytes(8, byteorder='big'))
        
        first_obs = episode[0].observation
        if hasattr(first_obs, 'tobytes'):
            obs_summary = first_obs.flatten()[:min(100, first_obs.size)]
            hasher.update(obs_summary.tobytes())
        else:
            hasher.update(str(first_obs).encode())
        
        actions = np.array([t.action for t in episode], dtype=np.int32)
        hasher.update(actions.tobytes())
        
        rewards = np.array([t.reward for t in episode])
        if rewards.size > 0:
            reward_summary = np.array([rewards.sum(), rewards.mean()], dtype=np.float32)
            hasher.update(reward_summary.tobytes())
        
        last_obs = episode[-1].next_observation
        if hasattr(last_obs, 'tobytes'):
            obs_summary = last_obs.flatten()[:min(100, last_obs.size)]
            hasher.update(obs_summary.tobytes())
        else:
            hasher.update(str(last_obs).encode())
        
        terminal_info = np.array([t.terminal for t in episode], dtype=bool)
        hasher.update(terminal_info.tobytes())
        
        hash_value = int(hasher.hexdigest(), 16)
        self._hash_cache[episode_id] = hash_value
        
        return hash_value

    def get_all_episodes(self):
        """全てのエピソードを取得（タイミング測定付き）"""
        start_time = time.time()
        
        result = []
        for episode in self.buffer:
            optimized_episode = []
            for t in episode:
                obs = t.observation
                if hasattr(t.observation, 'dtype') and t.observation.dtype != np.float32:
                    obs = np.array(t.observation, dtype=np.float32, copy=True)
                elif hasattr(t.observation, 'copy'):
                    obs = t.observation.copy()
                
                next_obs = t.next_observation
                if hasattr(t.next_observation, 'dtype') and t.next_observation.dtype != np.float32:
                    next_obs = np.array(t.next_observation, dtype=np.float32, copy=True)
                elif hasattr(t.next_observation, 'copy'):
                    next_obs = t.next_observation.copy()
                
                reward = t.reward
                if hasattr(t.reward, 'dtype') and t.reward.dtype != np.float32:
                    reward = np.array(t.reward, dtype=np.float32, copy=True)
                elif hasattr(t.reward, 'copy'):
                    reward = t.reward.copy()
                
                optimized_transition = Transition(
                    observation=obs,
                    action=t.action,
                    reward=reward,
                    next_observation=next_obs,
                    terminal=t.terminal
                )
                if hasattr(t, 'objective_values'):
                    optimized_transition.objective_values = t.objective_values
                if hasattr(t, 'solution_execution_time'):
                    optimized_transition.solution_execution_time = t.solution_execution_time
                
                optimized_episode.append(optimized_transition)
            result.append(optimized_episode)
        
        deserialization_start = time.time()
        self.buffer.clear()
        self.episode_hashes.clear()
        self._hash_cache.clear()
        deserialization_end = time.time()
        end_time = time.time()
        
        self.timing_stats['get_all_episodes_calls'] += 1
        self.timing_stats['get_all_episodes_time'] += (end_time - start_time)
        self.timing_stats['deserialization_time'] += (deserialization_end - deserialization_start)
        
        return result

    def get_timing_stats(self):
        """タイミング統計を取得"""
        return self.timing_stats

    def size(self):
        return len(self.buffer)


# =========================
# Actor (Ray Actor)
# =========================
@ray.remote
class Learner:
    """Learner（重みを提供するため）"""
    def __init__(self, config):
        self.config = config
        self.timing_stats = {
            'get_weights_calls': 0,
            'get_weights_time': 0.0,
            'weights_size': 0,
        }
        
        # 環境とエージェントの初期化（簡易版）
        job_generator = JobGenerator(
            0, 1,
            config['param_env']['n_window'],
            config['param_env']['n_on_premise_node'],
            config['param_env']['n_cloud_node'],
            config, N_JOBS, 0.2, 0
        )
        jobs_set = job_generator.generate_jobs_set()
        env = SchedulingEnvCacheOptimized(
            np.inf,
            config['param_env']['n_window'],
            config['param_env']['n_on_premise_node'],
            config['param_env']['n_cloud_node'],
            config['param_env']['n_job_queue_obs'],
            config['param_env']['n_job_queue_bck'],
            config['param_agent']['weight_wt'],
            config['param_agent']['weight_cost'],
            config['param_env']['penalty_not_allocate'],
            config['param_env']['penalty_invalid_action'],
            jobs_set,
            None, flag=0
        )
        
        self.agent = PCN(
            env,
            device='cpu',
            state_dim=env.observation_space.shape[0],
            scaling_factor=np.array([1, 1, 1]),
            learning_rate=1e-2,
            batch_size=512,
            hidden_dim=512,
            project_name="temp",
            experiment_name="PCN",
            log=False,
            debug_mode=False,
            use_enhanced_model=False,
        )

    def get_weights(self):
        """モデルの重みを取得（タイミング測定付き）"""
        start_time = time.time()
        weights = self.agent.model.state_dict()
        # CPUに移動してシリアライゼーション可能にする
        weights_cpu = {k: v.cpu() for k, v in weights.items()}
        end_time = time.time()
        
        # 重みのサイズを計算
        total_size = sum(v.nbytes for v in weights_cpu.values())
        
        self.timing_stats['get_weights_calls'] += 1
        self.timing_stats['get_weights_time'] += (end_time - start_time)
        self.timing_stats['weights_size'] = total_size
        
        return weights_cpu
    
    def get_weights_ref(self):
        """モデルの重みのObjectRefを取得（重みの共有用）"""
        if not hasattr(self, '_weights_ref') or self._weights_ref is None:
            # 初回のみ重みを取得してObject Storeに保存
            weights = self.get_weights()
            self._weights_ref = ray.put(weights)
        return self._weights_ref
    
    def update_weights_ref(self):
        """重みを更新してObjectRefを更新（学習後に呼び出す）"""
        weights = self.get_weights()
        self._weights_ref = ray.put(weights)
        return self._weights_ref

    def get_timing_stats(self):
        """タイミング統計を取得"""
        return self.timing_stats


@ray.remote
class Actor:
    def __init__(self, config, buffer, learner, actor_id=0):
        self.config = config
        self.actor_id = actor_id
        self.buffer = buffer
        self.learner = learner
        self.timing_stats = {
            'episode_generation_time': 0.0,
            'serialization_time': 0.0,
            'send_time': 0.0,
            'episodes_generated': 0,
            'weights_fetch_time': 0.0,
            'weights_fetch_calls': 0,
        }
        
        # 環境とエージェントの初期化
        job_generator = JobGenerator(
            0, 1,
            config['param_env']['n_window'],
            config['param_env']['n_on_premise_node'],
            config['param_env']['n_cloud_node'],
            config, N_JOBS, 0.2, 0
        )
        jobs_set = job_generator.generate_jobs_set()
        self.env = SchedulingEnvCacheOptimized(
            np.inf,
            config['param_env']['n_window'],
            config['param_env']['n_on_premise_node'],
            config['param_env']['n_cloud_node'],
            config['param_env']['n_job_queue_obs'],
            config['param_env']['n_job_queue_bck'],
            config['param_agent']['weight_wt'],
            config['param_agent']['weight_cost'],
            config['param_env']['penalty_not_allocate'],
            config['param_env']['penalty_invalid_action'],
            jobs_set,
            None, flag=0
        )
        
        self.agent = PCN(
            self.env,
            device='cpu',
            state_dim=self.env.observation_space.shape[0],
            scaling_factor=np.array([1, 1, 1]),
            learning_rate=1e-2,
            batch_size=512,
            hidden_dim=512,
            project_name="temp",
            experiment_name="PCN",
            log=False,
            debug_mode=False,
            use_enhanced_model=False,
        )

    def run(self, n_episodes=1, random_actions=True, fetch_weights=True):
        """エピソードを生成して送信（タイミング測定付き）"""
        collected_episodes = []
        
        # 重みの取得時間を測定（非ランダムアクションの場合）
        if not random_actions and fetch_weights:
            weights_fetch_start = time.time()
            # 重みのObjectRefを取得（共有された重みを使用）
            if not hasattr(self, '_weights_ref') or self._weights_ref is None:
                # 初回は重みのObjectRefを取得
                self._weights_ref = ray.get(self.learner.get_weights_ref.remote())
            # ObjectRefから重みを取得（RayのObject Storeから直接取得、シリアライゼーションは1回のみ）
            weights = ray.get(self._weights_ref)
            self.agent.model.load_state_dict(weights)
            # 目標値を設定（デフォルト値）
            self.agent.set_desired_return_and_horizon(np.array([0.0, 0.0]), 100.0)
            weights_fetch_end = time.time()
            
            self.timing_stats['weights_fetch_time'] += (weights_fetch_end - weights_fetch_start)
            self.timing_stats['weights_fetch_calls'] += 1
        
        for ep in range(n_episodes):
            # エピソード生成時間を測定
            episode_start = time.time()
            episode = self._run_episode(random_actions)
            episode_end = time.time()
            
            collected_episodes.append(episode)
            self.timing_stats['episode_generation_time'] += (episode_end - episode_start)
            self.timing_stats['episodes_generated'] += 1
        
        # シリアライゼーション時間を測定
        serialization_start = time.time()
        # エピソードデータのサイズを計算
        total_size = 0
        for episode in collected_episodes:
            for t in episode:
                if hasattr(t.observation, 'nbytes'):
                    total_size += t.observation.nbytes
                if hasattr(t.next_observation, 'nbytes'):
                    total_size += t.next_observation.nbytes
                if hasattr(t.reward, 'nbytes'):
                    total_size += t.reward.nbytes
        
        serialization_end = time.time()
        
        # 送信時間を測定
        send_start = time.time()
        ray.get(self.buffer.add_batch.remote(collected_episodes))
        send_end = time.time()
        
        self.timing_stats['serialization_time'] += (serialization_end - serialization_start)
        self.timing_stats['send_time'] += (send_end - send_start)
        
        return len(collected_episodes), total_size

    def _run_episode(self, random_actions):
        """エピソードを実行"""
        obs = self.env.reset()
        done = False
        transitions = []
        
        if random_actions:
            episode_seed = (int(time.time() * 1000000) + self.actor_id * 10000 + hash(str(obs))) % 10000
            np.random.seed(episode_seed)
        
        while not done:
            if random_actions:
                action = self.env.action_space.sample()
            else:
                action = self.agent.eval(obs)
            
            n_obs, reward, scheduled, wt_step, done = self.env.step(action)
            
            if hasattr(obs, 'dtype') and obs.dtype != np.float32:
                obs = np.array(obs, dtype=np.float32, copy=True)
            if hasattr(n_obs, 'dtype') and n_obs.dtype != np.float32:
                n_obs = np.array(n_obs, dtype=np.float32, copy=True)
            
            transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
            obs = n_obs
        
        if done:
            self.env.finalize_window_history()
            cost, _, avg_waiting_time = self.env.calc_objective_values()
            if len(transitions) > 0:
                transitions[0].objective_values = [cost, _, avg_waiting_time]
        
        return transitions

    def get_timing_stats(self):
        """タイミング統計を取得"""
        return self.timing_stats


# =========================
# メイン関数
# =========================
def main():
    print("="*60)
    print("Rayシリアライゼーション時間測定")
    print("="*60)
    
    # ReplayBufferを作成
    buffer = ReplayBuffer.remote(max_size=10000)
    
    # Learnerを作成
    learner = Learner.remote(config)
    
    # Actorを作成
    actors = [Actor.remote(config, buffer, learner, actor_id=i) for i in range(N_ACTORS)]
    
    # エピソード生成と送信の時間を測定
    print(f"\n{N_ACTORS}個のActorで各{EPISODES_PER_ACTOR}エピソードを生成中...")
    
    overall_start = time.time()
    
    # 各Actorでエピソードを生成（重みを取得する場合）
    print("\n重みを取得してエピソードを生成中...")
    futures = []
    for actor in actors:
        future = actor.run.remote(n_episodes=EPISODES_PER_ACTOR, random_actions=False, fetch_weights=True)
        futures.append(future)
    
    # 結果を収集
    results = []
    for future in futures:
        result = ray.get(future)
        results.append(result)
    
    overall_end = time.time()
    overall_time = overall_end - overall_start
    
    # エピソード取得の時間を測定
    print("\nReplayBufferからエピソードを取得中...")
    get_start = time.time()
    all_episodes = ray.get(buffer.get_all_episodes.remote())
    get_end = time.time()
    get_time = get_end - get_start
    
    # 統計情報を取得
    buffer_stats = ray.get(buffer.get_timing_stats.remote())
    learner_stats = ray.get(learner.get_timing_stats.remote())
    actor_stats_list = [ray.get(actor.get_timing_stats.remote()) for actor in actors]
    
    # 結果を表示
    print("\n" + "="*60)
    print("タイミング結果")
    print("="*60)
    
    print(f"\n--- 全体の時間 ---")
    print(f"総実行時間: {overall_time:.4f}秒")
    print(f"エピソード取得時間: {get_time:.4f}秒")
    
    print(f"\n--- Actor統計（合計） ---")
    total_episode_gen_time = sum(s['episode_generation_time'] for s in actor_stats_list)
    total_serialization_time = sum(s['serialization_time'] for s in actor_stats_list)
    total_send_time = sum(s['send_time'] for s in actor_stats_list)
    total_weights_fetch_time = sum(s['weights_fetch_time'] for s in actor_stats_list)
    total_weights_fetch_calls = sum(s['weights_fetch_calls'] for s in actor_stats_list)
    total_episodes = sum(s['episodes_generated'] for s in actor_stats_list)
    total_data_size = sum(r[1] for r in results)
    
    print(f"エピソード生成時間（合計）: {total_episode_gen_time:.4f}秒")
    print(f"シリアライゼーション時間（合計）: {total_serialization_time:.4f}秒")
    print(f"送信時間（合計）: {total_send_time:.4f}秒")
    print(f"重み取得時間（合計）: {total_weights_fetch_time:.4f}秒")
    print(f"重み取得回数: {total_weights_fetch_calls}")
    print(f"生成エピソード数: {total_episodes}")
    print(f"総データサイズ: {total_data_size / (1024*1024):.2f}MB")
    
    print(f"\n--- Learner統計 ---")
    print(f"get_weights呼び出し回数: {learner_stats['get_weights_calls']}")
    print(f"get_weights総時間: {learner_stats['get_weights_time']:.4f}秒")
    print(f"重みのサイズ: {learner_stats['weights_size'] / (1024*1024):.2f}MB")
    
    print(f"\n--- ReplayBuffer統計 ---")
    print(f"add_batch呼び出し回数: {buffer_stats['add_batch_calls']}")
    print(f"add_batch総時間: {buffer_stats['add_batch_time']:.4f}秒")
    print(f"get_all_episodes呼び出し回数: {buffer_stats['get_all_episodes_calls']}")
    print(f"get_all_episodes総時間: {buffer_stats['get_all_episodes_time']:.4f}秒")
    print(f"シリアライゼーション時間（合計）: {buffer_stats['serialization_time']:.4f}秒")
    print(f"デシリアライゼーション時間（合計）: {buffer_stats['deserialization_time']:.4f}秒")
    
    print(f"\n--- 時間の内訳 ---")
    print(f"エピソード生成時間: {total_episode_gen_time:.4f}秒 ({total_episode_gen_time/overall_time*100:.1f}%)")
    print(f"重み取得時間: {total_weights_fetch_time:.4f}秒 ({total_weights_fetch_time/overall_time*100:.1f}%)")
    print(f"Actor側シリアライゼーション時間: {total_serialization_time:.4f}秒 ({total_serialization_time/overall_time*100:.1f}%)")
    print(f"Actor→Buffer送信時間: {total_send_time:.4f}秒 ({total_send_time/overall_time*100:.1f}%)")
    print(f"Buffer側処理時間: {buffer_stats['add_batch_time']:.4f}秒 ({buffer_stats['add_batch_time']/overall_time*100:.1f}%)")
    print(f"Buffer→Learner取得時間: {get_time:.4f}秒 ({get_time/overall_time*100:.1f}%)")
    print(f"その他（オーバーヘッド）: {overall_time - total_episode_gen_time - total_weights_fetch_time - total_serialization_time - total_send_time - buffer_stats['add_batch_time'] - get_time:.4f}秒")
    
    print(f"\n--- データ転送速度 ---")
    if total_send_time > 0:
        send_speed = total_data_size / total_send_time / (1024*1024)  # MB/s
        print(f"Actor→Buffer転送速度: {send_speed:.2f}MB/s")
    if get_time > 0:
        get_speed = total_data_size / get_time / (1024*1024)  # MB/s
        print(f"Buffer→Learner転送速度: {get_speed:.2f}MB/s")
    if total_weights_fetch_time > 0 and learner_stats['weights_size'] > 0:
        weights_speed = (learner_stats['weights_size'] * total_weights_fetch_calls) / total_weights_fetch_time / (1024*1024)  # MB/s
        print(f"Learner→Actor重み転送速度: {weights_speed:.2f}MB/s")
    
    print(f"\n--- エピソードあたりの時間 ---")
    if total_episodes > 0:
        print(f"エピソード生成時間/エピソード: {total_episode_gen_time/total_episodes:.4f}秒")
        print(f"送信時間/エピソード: {total_send_time/total_episodes:.4f}秒")
        print(f"データサイズ/エピソード: {total_data_size/total_episodes/(1024*1024):.2f}MB")
    
    print("\n" + "="*60)
    
    # 詳細なActor統計
    print("\n--- 各Actorの詳細統計 ---")
    for i, (stats, result) in enumerate(zip(actor_stats_list, results)):
        print(f"\nActor {i}:")
        print(f"  エピソード生成時間: {stats['episode_generation_time']:.4f}秒")
        print(f"  重み取得時間: {stats['weights_fetch_time']:.4f}秒")
        print(f"  シリアライゼーション時間: {stats['serialization_time']:.4f}秒")
        print(f"  送信時間: {stats['send_time']:.4f}秒")
        print(f"  生成エピソード数: {stats['episodes_generated']}")
        print(f"  データサイズ: {result[1]/(1024*1024):.2f}MB")
        if stats['send_time'] > 0:
            print(f"  転送速度: {result[1]/stats['send_time']/(1024*1024):.2f}MB/s")


if __name__ == "__main__":
    main()


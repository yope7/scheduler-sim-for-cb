export const meta = {
  name: 'trace24-pcn-goal-loop',
  description: 'Trace24 PCN: best-of-3 train -> best-checkpoint eval -> judge -> tuner, loop until wide+many PF goal',
  whenToUse: 'Find a trace24 PCN config that reliably yields a wide, dense Pareto front (unique>=32 & passed).',
  phases: [
    { title: 'Train', detail: 'best-of-3 PCN trainings in parallel (Ray, no ray-stop mid-flight)' },
    { title: 'Eval', detail: 'best-checkpoint eval (basic gate) per run' },
    { title: 'Tune', detail: 'diagnose collapse, propose next 3 configs' },
  ],
}

// ---- fixed paths / base env ----
const ROOT = '/home/noguchi/scheduler-sim-for-cb'
const CONFIG = 'experiments/distributed_pcn/job_trace_24_scratch_pass.yml'
const REF = 'experiments/distributed_pcn/trace24_no_outlier_sampled_exhaustive_pf.npz'
const PY = '.venv/bin/python'

// /goal
const GOAL_UNIQUE = 32
const GOAL_MEAN_GAP = 90
const MAX_ROUNDS = 10
const N_ITER_DEFAULT = (args && args.n_iter) || 60

function envStr(cfg) {
  const e = cfg.env || {}
  const kv = {
    PCN_CHOOSE_COMMANDS_MODE: e.PCN_CHOOSE_COMMANDS_MODE || 'pf_archive',
    PCN_TRAIN_KNEE_PF_WEIGHT: e.PCN_TRAIN_KNEE_PF_WEIGHT || '8',
    PCN_TRAIN_LOW_SLOPE_PF_WEIGHT: e.PCN_TRAIN_LOW_SLOPE_PF_WEIGHT || '6',
    PCN_TRAIN_LOW_WAIT_PF_WEIGHT: e.PCN_TRAIN_LOW_WAIT_PF_WEIGHT || '10',
  }
  // optional conditioning knobs only if provided
  for (const k of ['PCN_CONDITIONING_SENS_WEIGHT', 'PCN_CONDITIONING_KL_MARGIN', 'PCN_COND_ADD_SCALE', 'PCN_S_EMB_DROPOUT']) {
    if (e[k] !== undefined && e[k] !== null && `${e[k]}` !== '') kv[k] = `${e[k]}`
  }
  return Object.entries(kv).map(([k, v]) => `${k}=${v}`).join(' ')
}

function trainCmd(cfg, round, idx) {
  const out = `experiments/distributed_pcn/wf_r${round}_i${idx}`
  const nIter = cfg.n_iter || N_ITER_DEFAULT
  return `cd ${ROOT} && OUT=${out} && rm -rf $OUT && mkdir -p $OUT && \
DISTRIBUTED_PCN_CONFIG=${CONFIG} DISTRIBUTED_PCN_OUTPUT_DIR=$OUT \
DISTRIBUTED_PCN_SUPERVISED_EPOCHS=0 DISTRIBUTED_PCN_N_ITERATIONS=${nIter} \
PCN_TRAIN_LOW_WAIT_MAX=600 PCN_EVAL_GAP_BOOST_MAX=5.0 DISTRIBUTED_PCN_CMD_OUTCOMES=1 \
${envStr(cfg)} \
PYTHONUNBUFFERED=1 ${PY} -u -m src.distributed.distributed_pcn_event --conditioning --mid-core --no-viz \
  > $OUT/train.log 2>&1 ; echo "TRAIN_EXIT=$?" ; \
EXEC=$(find $OUT -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1) ; \
echo "EXEC_DIR=$EXEC" ; echo "N_CKPT=$(find $EXEC -name 'model_iter_*.pth' 2>/dev/null | wc -l)"`
}

function evalCmd(cfg, execDir) {
  const e = cfg.env || {}
  const grid = e.EVAL_GRID || '32'
  const cmdMode = e.PCN_EVAL_COMMAND_MODE || 'pf_ref'
  const lwQuota = e.PCN_PF_COMMAND_LOW_WAIT_QUOTA || '8'
  const lwFrac = e.PCN_PF_COMMAND_LOW_WAIT_FRAC || '0.35'
  const topk = e.BEST_CKPT_TOPK || '3'
  return `cd ${ROOT} && EXEC_DIR='${execDir}' REF_NPZ='${REF}' PCN_REF_PF_NPZ='${REF}' \
DISTRIBUTED_PCN_CONFIG=${CONFIG} DISTRIBUTED_PCN_USE_EVENT_NATIVE=1 PYTHONPATH=. \
EVAL_GRID=${grid} PCN_EVAL_COMMAND_MODE=${cmdMode} \
PCN_SCORE_LOW_WAIT_MAX=0 PCN_SCORE_MIN_LOW_WAIT_COVERED_FRAC=0 \
PCN_PF_COMMAND_LOW_WAIT_MAX=600 PCN_PF_COMMAND_LOW_WAIT_FRAC=${lwFrac} PCN_PF_COMMAND_LOW_WAIT_QUOTA=${lwQuota} \
BEST_CKPT_TOPK=${topk} BEST_CKPT_MIN_ITER_FRAC=0.3 \
${PY} -u -m scripts.eval_best_checkpoint > $EXEC_DIR/besteval.log 2>&1 ; echo "EVAL_EXIT=$?" ; \
echo '--- pf_score.json ---' ; cat $EXEC_DIR/pf_score.json ; \
echo '--- best_checkpoint_selection.json ---' ; cat $EXEC_DIR/best_checkpoint_selection.json ; \
echo '--- archive series ---' ; ${PY} -c "import json,glob; f=glob.glob('$EXEC_DIR/training_iteration_summary.json'); d=json.load(open(f[0])) if f else {'rows':[]}; print(','.join(str(r.get('pareto_front_size')) for r in d.get('rows',[]) if r.get('iteration',0)%10==0))"`
}

const TRAIN_OUT = {
  type: 'object',
  properties: {
    exec_dir: { type: 'string', description: 'EXEC_DIR printed by the command (timestamped subdir with checkpoints)' },
    n_checkpoints: { type: 'integer' },
    train_exit: { type: 'integer' },
  },
  required: ['exec_dir', 'n_checkpoints', 'train_exit'],
}

const EVAL_OUT = {
  type: 'object',
  properties: {
    exec_dir: { type: 'string' },
    unique_n: { type: 'integer', description: 'eval_pf_unique_n from pf_score.json' },
    passed: { type: 'boolean' },
    mean_gap: { type: 'number' },
    frac_bad: { type: 'number' },
    cost_span: { type: 'number' },
    selected_iter: { type: 'integer' },
    archive_series: { type: 'string', description: 'comma-sep pareto_front_size every 10 iters (collapse signature)' },
    eval_ok: { type: 'boolean', description: 'true if pf_score.json was produced' },
  },
  required: ['unique_n', 'passed', 'mean_gap', 'frac_bad', 'cost_span', 'eval_ok'],
}

const NEXT_CONFIGS = {
  type: 'object',
  properties: {
    diagnosis: { type: 'string' },
    configs: {
      type: 'array',
      minItems: 3,
      maxItems: 3,
      items: {
        type: 'object',
        properties: {
          label: { type: 'string' },
          n_iter: { type: 'integer' },
          PCN_CONDITIONING_SENS_WEIGHT: { type: 'string' },
          PCN_COND_ADD_SCALE: { type: 'string' },
          PCN_CHOOSE_COMMANDS_MODE: { type: 'string' },
          PCN_EVAL_COMMAND_MODE: { type: 'string' },
          EVAL_GRID: { type: 'string' },
          PCN_PF_COMMAND_LOW_WAIT_QUOTA: { type: 'string' },
          PCN_PF_COMMAND_LOW_WAIT_FRAC: { type: 'string' },
          PCN_TRAIN_LOW_WAIT_PF_WEIGHT: { type: 'string' },
          PCN_TRAIN_LOW_SLOPE_PF_WEIGHT: { type: 'string' },
          PCN_TRAIN_KNEE_PF_WEIGHT: { type: 'string' },
          BEST_CKPT_TOPK: { type: 'string' },
        },
        required: ['label'],
      },
    },
  },
  required: ['diagnosis', 'configs'],
}

function toCfg(o) {
  const env = {}
  for (const k of ['PCN_CONDITIONING_SENS_WEIGHT', 'PCN_COND_ADD_SCALE', 'PCN_CHOOSE_COMMANDS_MODE',
    'PCN_EVAL_COMMAND_MODE', 'EVAL_GRID', 'PCN_PF_COMMAND_LOW_WAIT_QUOTA', 'PCN_PF_COMMAND_LOW_WAIT_FRAC',
    'PCN_TRAIN_LOW_WAIT_PF_WEIGHT', 'PCN_TRAIN_LOW_SLOPE_PF_WEIGHT', 'PCN_TRAIN_KNEE_PF_WEIGHT', 'BEST_CKPT_TOPK']) {
    if (o[k] !== undefined && o[k] !== null && `${o[k]}` !== '') env[k] = `${o[k]}`
  }
  return { label: o.label || 'cfg', n_iter: o.n_iter || N_ITER_DEFAULT, env }
}

function goalMet(m) {
  return !!m && m.eval_ok && m.passed && (m.unique_n >= GOAL_UNIQUE) && (m.mean_gap <= GOAL_MEAN_GAP)
}

function rankKey(m) {
  if (!m || !m.eval_ok) return [-1, -1, 0]
  return [m.passed ? 1 : 0, m.unique_n || 0, -(m.frac_bad ?? 1)]
}

function better(a, b) {
  const ka = rankKey(a), kb = rankKey(b)
  for (let i = 0; i < ka.length; i++) { if (ka[i] !== kb[i]) return ka[i] > kb[i] ? a : b }
  return a
}

// round-0 configs: baseline + 2 variants (conditioning sens / command mode)
let configs = [
  toCfg({ label: 'baseline' }),
  toCfg({ label: 'sens0.06', PCN_CONDITIONING_SENS_WEIGHT: '0.06' }),
  toCfg({ label: 'pfmixed-grid40', PCN_EVAL_COMMAND_MODE: 'pf_ref', EVAL_GRID: '40', BEST_CKPT_TOPK: '4' }),
]

let best = null

for (let round = 1; round <= MAX_ROUNDS; round++) {
  phase('Train')
  log(`Round ${round}: training ${configs.length} configs (${configs.map(c => c.label).join(', ')}) at n_iter=${configs.map(c => c.n_iter).join('/')}`)

  const results = await pipeline(
    configs,
    // stage 1: train
    (cfg, _orig, idx) => agent(
      `Run EXACTLY this bash command to train a PCN model. Use the Bash tool with timeout 600000 (10 min max). ` +
      `Do not modify the command. When it finishes, parse the printed lines TRAIN_EXIT=, EXEC_DIR=, N_CKPT= and return them. ` +
      `If EXEC_DIR is empty or N_CKPT is 0, still return what you have (train_exit may be nonzero / timeout).\n\nCOMMAND:\n${trainCmd(cfg, round, idx)}`,
      { label: `train:r${round}.${configs[idx].label}`, phase: 'Train', schema: TRAIN_OUT }
    ),
    // stage 2: eval (prev = TRAIN_OUT)
    (tr, cfg, idx) => {
      if (!tr || !tr.exec_dir || tr.n_checkpoints < 1) return Promise.resolve({ exec_dir: (tr && tr.exec_dir) || '', unique_n: 0, passed: false, mean_gap: 9999, frac_bad: 1, cost_span: 0, eval_ok: false, archive_series: '' })
      return agent(
        `Run EXACTLY this bash command to evaluate PCN checkpoints (best-checkpoint selection, basic gate). ` +
        `Use the Bash tool with timeout 600000. Do not modify it. It prints pf_score.json and best_checkpoint_selection.json and an archive series. ` +
        `Return: unique_n=eval_pf_unique_n, passed, mean_gap, frac_bad, cost_span, selected_iter from pf_score.json; archive_series from the '--- archive series ---' line; eval_ok=true if pf_score.json was printed (else false). exec_dir='${tr.exec_dir}'.\n\nCOMMAND:\n${evalCmd(configs[idx], tr.exec_dir)}`,
        { label: `eval:r${round}.${configs[idx].label}`, phase: 'Eval', schema: EVAL_OUT }
      )
    }
  )

  const valid = results.filter(Boolean)
  for (const m of valid) best = better(m, best)
  const top = valid.reduce((a, b) => better(b, a), null)
  log(`Round ${round} results: ` + valid.map(m => `${m.unique_n}pt${m.passed ? '/pass' : ''}(gap${Math.round(m.mean_gap)})`).join('  ') +
    `  | best-so-far: ${best ? `${best.unique_n}pt${best.passed ? '/pass' : ''} gap${Math.round(best.mean_gap)}` : 'none'}`)

  if (goalMet(best)) {
    log(`GOAL MET at round ${round}: ${best.unique_n} pts, passed, mean_gap ${Math.round(best.mean_gap)}. exec=${best.exec_dir}`)
    break
  }
  if (round === MAX_ROUNDS) {
    log(`MAX_ROUNDS reached. best: ${best ? `${best.unique_n}pt passed=${best.passed} gap=${Math.round(best.mean_gap)} exec=${best.exec_dir}` : 'none'}`)
    break
  }

  // tuner: diagnose & propose next 3 configs
  phase('Tune')
  const summary = valid.map((m, i) => `run ${i} (${configs[i] ? configs[i].label : '?'}): unique=${m.unique_n} passed=${m.passed} mean_gap=${m.mean_gap?.toFixed(1)} frac_bad=${m.frac_bad?.toFixed(2)} cost_span=${Math.round(m.cost_span)} selected_iter=${m.selected_iter} archive_series=[${m.archive_series}]`).join('\n')
  const tuned = await agent(
    `You tune a PCN (Pareto Conditioned Networks) scheduler to maximize the eval Pareto-front: GOAL = unique_n>=${GOAL_UNIQUE} AND passed AND mean_gap<=${GOAL_MEAN_GAP} (basic gate). ` +
    `The core failure is Phase3 policy mode-collapse: archive_series (pareto_front_size every 10 iters) crashes to single digits and the eval PF shrinks. ` +
    `best-checkpoint eval already picks the richest checkpoint, so focus on configs that (a) avoid late collapse and (b) widen/denseify the PF. ` +
    `Knobs you may set (strings): PCN_CONDITIONING_SENS_WEIGHT (0.03 default; 0.05/0.08 raises command sensitivity to fight collapse), PCN_COND_ADD_SCALE (0.25 default; 0.35), ` +
    `PCN_CHOOSE_COMMANDS_MODE (pf_archive/pf_ref/pf_mixed), PCN_EVAL_COMMAND_MODE (pf_ref/pf_archive), EVAL_GRID (32/40/48 -> more eval points), ` +
    `PCN_PF_COMMAND_LOW_WAIT_QUOTA/FRAC (more low-wait commands -> wider left tail), PCN_TRAIN_LOW_WAIT_PF_WEIGHT/LOW_SLOPE/KNEE (replay weights), ` +
    `BEST_CKPT_TOPK (more checkpoints evaluated, default 3 -> 5), n_iter (default ${N_ITER_DEFAULT}; raising helps late recovery but each run must stay < ~9 min). ` +
    `Current best-so-far: ${best ? `${best.unique_n}pt passed=${best.passed} gap=${best.mean_gap?.toFixed(1)}` : 'none'}.\n\n` +
    `This round's runs:\n${summary}\n\n` +
    `Propose EXACTLY 3 next configs (keep your best-performing settings, vary 1-2 knobs each to explore toward the goal). Give a short diagnosis.`,
    { label: `tune:r${round}`, phase: 'Tune', schema: NEXT_CONFIGS }
  )
  log(`Round ${round} tuner: ${tuned.diagnosis}`)
  configs = tuned.configs.map(toCfg)
}

return {
  goal_met: goalMet(best),
  best: best ? {
    exec_dir: best.exec_dir, unique_n: best.unique_n, passed: best.passed,
    mean_gap: best.mean_gap, frac_bad: best.frac_bad, cost_span: best.cost_span,
    selected_iter: best.selected_iter,
  } : null,
}

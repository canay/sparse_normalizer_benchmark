Model: claude-opus-5-5

# Narrow V8 engineering review: matched bottom-k policy control (MCH-SND-007, source revision 9)

- **Run:** snd-matched-bottomk-engineering-20260928-v8. **Operation:** snd-approved-b-repair-20260928.
- **Reviewer:** claude-opus-5-5, requested effort max (no observed effort field). Latest clock read 2026-09-28 11:47 +03:00 (08:47Z).
- **Read-only.** No writes, network, transfer, packet/fixture imports or execution, training, or process actions. Bash/Python ran only my own in-memory hash, AST, JSON and glob census code.
- **Selected scope:**
  - the 8 V7→V8 delta paths;
  - their callers: `launch.py` routes, `analyze.py` L89/L124, `supervision_coverage`, `engineering_verify` smoke census, `process_fault_checks` monitor/fault paths, `snapshot_boundary`, the worker handoff, and lineage `manifest-check`;
  - proof, state and capsule custody.
- This is not manuscript Round B, not a novelty/venue certificate, and not a compute, acquisition, smoke or main release.

## 1. Verdict: CODE_ADMISSIBLE_PENDING_RUNTIME_ADMISSION

- **P1: none. P2: none.** I found no concrete, reachable path in the V8 delta that harms registered-50 integrity or availability.
- **V7-1 (the prior P2) is closed.** The producer now always publishes five artifacts, and prior sessions are checked with the shared predicate before any new attempt. V7-2 through V7-5 are implemented as specified.
- **Why availability cannot get worse:**
  - The shared predicate `validated_supervision_terminal` is the V7 delivery predicate minus one redundant containment line.
  - So every folder that V8 now latches on at controller start, smoke admission or analysis was already undeliverable in V7.
  - V8 only moves that failure earlier, before compute is spent. It adds no new way to lose the registered campaign.
  - The remaining changes remove the V7-1 early-close loss path, or refuse before any folder exists (identity/latch mismatch, signal before launch).
- **Integrity:** no path admits fallback telemetry as monitoring, heartbeat, attempt or scientific evidence.
- **Limits:** this admits code only. All runtime proofs in Section 5 remain open, and V7's ENGINEERING_REPAIR_REQUIRED stays on record as history.

### Non-blocking P3 notes (no repair campaign implied)

- **P3-a (hygiene):** `ENGINEERING_PLAN.md` L15 `resume_command` still shows the non-detached `supervise` form.
  - A literal copy fails closed before mkdir (`supervisor.py` L75-76, MAIN_SUPERVISOR_DETACHED_SESSION_REQUIRED) and creates no folder.
  - Align it with L89/L143 when the plan is next touched.
- **P3-b (runtime proof item, pre-existing):** after SIGKILL, `stop_verified_records` allows about 2 s before recording survivors (`support.py` L562-578).
  - A CUDA worker still tearing down at a supervisor-initiated resumable stop would become a survivor.
  - V8 then latches the whole campaign at the next start (`support.py` L209-215, `controller.py` L170-177). V7 would have lost it at delivery.
  - The outcome is the same, with less wasted compute. The Linux fault proof should measure real GPU-worker teardown against this window.
- **P3-c (informational):** a SIGTERM sent to the controller alone, with no supervisor-recorded cause, yields a code-3 `RuntimeError:CONTROLLER_SIGNAL_15` terminal. Delivery can finalize it only under an explicit parent release, and the result is non-evidence either way.
- **P3-d (informational):** the shared predicate drops V7 delivery's `resolve().relative_to()` containment check. The exact five-name set makes it redundant except for symlinks, which no producer creates.

## 2. Actual read depth

- **Owners, all read FULL this session with no context compaction:**
  - the installed akis-audit SKILL.md;
  - routes, bridge, START, RUN, RUN_ROUTE_INDEX;
  - Akis2 protocol, roles 00-04, rubric;
  - GATES, MANUSCRIPT_AUDIT_REVISION, METHOD_RIGOR, REFERENCE_INTEGRITY_GATE;
  - STUDY_DESIGN, METHODOLOGY_CHANGE_CONTROL, EXPERIMENT_EVIDENCE, EXPERIMENT_DURABILITY;
  - TOOL_ENVIRONMENT, DEVICE_REGISTRY, MODEL_ROUTING, CROSS_TOOL_CONTEXT_CONTRACT;
  - OPERATIONAL_MEMORY_STANDING, QUALITY_LOCKED_CONTEXT_MINIMAL, operational_memory INDEX.md;
  - the four shared audit references and the six named modules.
  - All 37 hashes equal the V8 bindings.
- **Route-directed extras, FULL:** project PROJECT_RULES.md, PIPELINE_POINTER, AGENTS, CLAUDE.md and START.
- **Not loaded (if the root rules these mandatory for this review, admission fails closed on them):**
  - The capsule's own producer-route (`manuscript_revision`) full-read set, hash-verified only:
    - `main.tex`, CS_CE_IS standard, FIGURE_TABLE_VISUAL_QA, LATEX_BUILD_POLICY, METHODOLOGY_OVERVIEW_FIGURE;
    - MANUSCRIPT_QUALITY_PROFILES/CONTRACT and the akis-writing/akis-revise references.
    - Reason: they belong to the producer's revision route, and this task excludes whole-manuscript reading.
  - Also not loaded:
    - CROSS_TOOL_REVIEW.md, LESSONS_LEARNED, ENVIRONMENT_NOTES and TOOL_RUN_LOG;
    - full HANDOFF and QUALITY_LEDGER (their validated AKIS-CURRENT blocks were used instead).
- **Code FULL:**
  - supervisor, support, controller, delivery, data_env_preflight, engineering_verify;
  - all 8 diffs;
  - callers launch, analyze, process_fault_checks, snapshot_boundary, worker;
  - `experiment_lineage_preflight` L270-389 only;
  - plan, matrix and STATUS.
- **Proof and state FULL:**
  - v8_focused_checks.py and its receipt; v8_declaration.py and its output;
  - the V8 transaction receipt, input manifest, checkpoint, parent admission and prelaunch binding;
  - frozen KUNYE (1202 lines), MCH ledger, the capsule fields and the four AKIS-CURRENT blocks;
  - the V7 review and its receipt.
- **Hash/AST only:** the other unchanged sources.

## 3. Focused adjudication (code and callers, not fixture labels)

**V7-1, normal-close producer and prior-session predicate: PASS.**
- **Close sequence:**
  - An early `poll()` exit and a first-iteration exception both reach cleanup, `wait` and log fsync.
  - `terminal_artifacts` (`supervisor.py` L45-54, called at L224) then creates only what is missing:
    - `samples.jsonl` via exclusive create (`xb`) plus fsync;
    - `heartbeat.json` carrying the actual token, cause, time, stop events and signals, with `terminal_fallback=true` and `monitoring_evidence=false`.
  - It then returns exactly `SUPERVISION_ARTIFACTS`. If `launch.json` or a log file is absent, `sha()` raises, no terminal is written, and the existing loss latch applies.
  - `Path.glob` on a not-yet-created campaign folder returns `[]` (measured on Python 3.12.12, with a positive control), so an early exit in a first session still closes.
- **The fallback can never count as evidence:**
  - authenticated handoff rejects it (`support.py` L171);
  - its samples are empty, so coverage cannot admit it (pulses must appear in samples, L333);
  - no attempt can exist without a real, authenticated heartbeat.
- **Shared predicate** (L103-129) checks:
  - status;
  - session, mode and campaign identity;
  - source and config hashes;
  - checked cleanup with `survivors==[]`;
  - chronology;
  - exact five names, by both length and set, with byte SHA and size;
  - the Popen no-controller fields.
- **Before any new attempt:**
  - `missing_supervision_terminals` applies the predicate to every non-current folder (L201-215).
  - The controller runs it inside its try block, before any cell, and writes SUPERVISION_LOSS_NON_EVIDENCE if anything fails.
  - Only the authenticated current folder is excluded (L202).
- **One predicate, no duplicated logic:**
  - delivery uses the same function for the latest-genuine, later-session and full-census checks (`delivery.py` L79, L93, L109);
  - `supervision_coverage` (Linux verifier and Windows analyzer) and `analyze.py` L89 inherit it.
- **Fault-test integration (by code reading):** in `process_fault_checks` monitor_case, fault-rss and fault-stale write the real `final_config` and source hashes, five artifacts, and zero survivors after the dummy processes are killed. Their pre-mkdir identity check passes because monitor_case creates the campaign folder without an `identity.json`. Smoke admission therefore still closes.

**V7-2, finality: PASS.**
- Delivery (L79-90) refuses when the latest genuine session's supervisor cause and all its observed causes are controlled-infrastructure causes. It allows the session only if one of these holds:
  - a resource or loss latch exists;
  - the controller classification is exactly `RuntimeError:INCOMPLETE_NON_EVIDENCE_MAXIMUM_TOTAL_ATTEMPTS` (`controller.py` L222/L249);
  - the parent release, stored outside the producer roots, carries `resume_forbidden=true`, a reason, and `controller_terminal_sha256==sha(final)`.
- Return codes None, 0, 75 and 76 remain refused (L58). The producer cannot author its own release.

**V7-3, pre-mkdir checks and genuine-latest selection: PASS.**
- `campaign_entry_identity` (`support.py` L81-93) runs read-only before mkdir (`supervisor.py` L80, versus mkdir at L110). The controller uses the same constructor (L153).
- An existing latch or an order mismatch leaves no folder (focused prelaunch cases).
- Latest selection uses only sessions with a bound CONTROLLER_TERMINAL (`delivery.py` L62-78).
- Later sessions must all show (L91-99):
  - a valid terminal;
  - zero live attempt references, `launched_attempt_count` 0 and empty refs;
  - a genuine fallback heartbeat if a controller started.
- Monitored, attempted, unclean or unclosed later sessions are refused.
- Only `ProcessLaunchFailure` is caught (L221), so post-fork exceptions stay unclosed and latch.

**V7-4, signals before launch: PASS.**
- Signal handlers are installed (L105-107) and `if stopping` is checked (L108), both before mkdir (L110).
- `launch_failure_terminal` carries the recorded signals, events and causes, and invents none.
- `administrative_nonexecution` requires `signals==[]` and `causes==[cause]` (`engineering_verify.py` L77-79), so signal-bearing smoke extras are refused.

**V7-5, cheap consistency items: PASS.**
- Exact length-plus-set checks are in coverage (`support.py` L304) and the verifier (L44, L92).
- The data-env planned command (L98) matches the plan's L89 template byte for byte. `manifest-check` reads only the platform envelope (L285-304), so the changed command string cannot block lineage admission.
- Frozen KUNYE blocking_items (L72-78) and STATUS.md now name V8.

## 4. Custody measured this session

- **Source:**
  - SOURCE_MANIFEST 2B6A7A75842F5A4DCA34B50B013CE1C98B63A727150D0C3E1D04DAF0C09E5E60, revision 9, predecessor CF6C72A1 (equal to the V7 snapshot manifest).
  - 40/40 rows match SHA and bytes both live and in `source_snapshots/v8`, with no unlisted members. 22/22 Python files parse.
  - Exactly the 8 stated paths changed from V7 to V8; the 8 diffs match the checkpoint's SHA and bytes; the V7 snapshot is 40/40.
- **Unchanged science (identical to V7):**
  - config 0876F92A, protocol 9A1A674D, benchmark EE2C6076, datasets C0D2E439, analyzer FA939D28;
  - worker, train_only, launch, acquire_archive and DECISION;
  - P5 proof 234C9BB4 is present;
  - MCH ledger 7EEA6F32 is unchanged since V7 (ends at the V5 entry plus lock 9a1a674d).
- **Packet:**
  - input manifest 1FF9B79D: 29/29 files; its 40 source rows equal SOURCE_MANIFEST; 37/37 owner bindings match;
  - checkpoint 19CEB295: 30/30 references.
- **Focused receipt:**
  - 1CB64AE1 lists 21 checks.
  - Producer C6230C13 is the current `v8_focused_checks.py`, and its six source bindings equal the current modules.
  - All 20 fixture trees carry the V7 manifest CF6C72A1 as scaffolding: the receipt was written at 11:19:38 +03:00, before the V8 freeze at 11:20:52.
  - This proves bounded current control flow only. It is not authentic source verification.
- **Declaration:** DECLARATION_PREFLIGHT_V8 17AB5305, producer DE33B770 (current); 40 sources/22 AST; durability and argument issues `[]`; PRELAUNCH rc0; binds 1CB64AE1.
- **State transaction 724004F5:**
  - Order was: advance two blocks, update_project_state, log_event, state_sync capture/check, then capsule build/validate.
  - Disclosed: state_sync check returned rc0 but printed a FAIL for the portfolio-refresh lock (outside the project), plus a vault WARN.
- **Guard snapshot:**
  - Guard manifest 63A4E3E7: 34/34 snapshots equal the raw input census, and all 34 appear in capsule 18D59784 with the same SHA and bytes. 7/7 frozen state files match.
  - The capsule is valid, with no fail-closed reasons; its 736,541 B budget warning is informational.
  - Timeline: 08:23:20Z receipt, 08:26:02Z snapshot, 08:31:07Z prelaunch binding, 08:31:23Z guard.
- **Retained records:** preimages 5B2BEF8E 44/44; temp-file census 19/19.
- **No results exist:** there are no local `supervision/`, `main/`, `engineering_smoke/`, `admission/`, `outputs/` or `data/` trees.
- **Live drift:** HANDOFF, KUNYE and QUALITY_LEDGER changed at 11:46 +03:00, when the root recorded the running guard and payload preparation. That is later, outside-packet history, not current production; this review binds the frozen snapshot.

## 5. Unexecuted proofs (none executed here; nothing above is runtime evidence)

- **Host:** a fresh owned MTA host, with resources and path captured at acquisition/start against disk ≥10 GiB and VRAM ≥3 GiB. The ~8 GB reading is frozen history; the 13.637 GiB reading is capacity only, not admission.
- **Data and environment:** archive SHA and size; train-only member order, arrays, splits and class counts; environment and lineage outputs.
- **Linux fault receipts:** PDEATHSIG, timeout 124, deadline, missing and stale heartbeat, owned and orphan cleanup, and the new producer close under real processes, including GPU teardown timing (P3-b).
- **Non-registered five-method runs, seeds 1000/1001:** ≤100 s and ≤4096 MB per cell; initial-weight parity; whole-epoch replay.
- **Interruption and resume:**
  - a GPU-observed ≥10 s midcell 75, resumed more than 600 s later as attempt-002, with a bitwise epoch-log match;
  - a boundary 75, with the first cell complete and immutable and the second not started;
  - advancing same-worker CPU over the whole interval.
- **Detached launch:** actual setsid/nohup, SID/TTY, PID tokens and unique noclobber logs.
- **Release and delivery chain:**
  - separate parent smoke and main releases;
  - the all-50, control and scalar verification;
  - parent transport, producer inventory, manual exact local delivery, final local release, then analyze.
- **Execution rules:** Linux runs go only through `launch.py`; the Windows analyzer and delivery run directly; no unowned IsaacSim and no privileged logs.
- matched50 remains unobserved, and the original adverse scientific decisions stand.

## 6. Boundaries

- Engineering code review only: no manuscript, venue, novelty, author-approval or compute-release verdict.
- These engineering findings must not enter a blind manuscript Round B.

ROUND_OUTPUT_COMPLETE
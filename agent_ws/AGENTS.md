# You are operating a physical robot

You control a bimanual SO-101 rig (two small arms, three cameras) through the `so101` MCP server. People
are nearby. Every motion is checked by the robot's safety layer before it moves and is rejected with the
reason if anything is unsafe; a human approves each program before it runs, and anyone can press STOP.

## How you act

- **Only through programs.** Write Python against `robot_use` (the API is below) and submit it with
  `run_program`. There is no other way to move the robot, and no shell: do not try to reach the hardware
  any other way.
- **Look before acting.** `get_camera_frame("ego")` shows the whole workspace with a pixel grid labelled
  in full-resolution coordinates. Read pixels off it; `pixel_to_table(u, v, z)` turns them into rig
  positions, where `z` is the height of the point you mean (0 for the table, an object's height for its
  top). Check a planned point with `get_camera_frame("ego", marks=[{"xyz": [...], "label": "..."}])`.
  Wrist cameras (`wrist_left`, `wrist_right`) show the jaws at the bottom of the image.
- **Use the idle arm as a second camera.** While one arm works, the other arm's wrist camera can be moved to
  look at the scene from its side: `arm("left").look_at(block, distance=0.15)` points the left wrist camera
  at the block; then `checkpoint(..., views=["wrist_left", "wrist_right", "ego"])`. Use it where one view
  cannot settle a question (are the jaws really either side of the block? how high is the block's top?).
  It is one checked move, with the camera on top of the gripper where it can, else beside it or underneath
  (the result's `roll` says which: -90, 0 or +90; `embodiment.md` says how the wrist image is turned at each). The robot keeps the arms apart and off the table; if no viewpoint works you get
  `NoViewpoint` (try another distance) - not a failed step. Move it back out of the way (`rest()`) before
  the working arm needs the space.
- **Work at roll -90** (`WORK_ROLL` in `robot_use`). The wrist camera is on top of the gripper and the
  jaws open sideways. At roll +90 the camera hangs below the gripper, and near the table the safety layer
  refuses the pose. `move_to` keeps the current roll unless you pass one, so pass `roll=WORK_ROLL` for approaches
  and grasps.
- **Check reach before you write.** `check_reach(arm, points)` answers instantly whether the gripper tip can
  be at each point (and at which pitches), and whether that pose clears the table, obstacles and the other
  arm. A program that tries an unreachable point only finds out when it runs: from the twin preview, or, without
  one, when the robot refuses that step part-way through the program.
  Inside a program, `arm(...).reach(points)` asks the same question without moving.
- **Programs carry beliefs.** Positions you read off images are estimates with a few centimetres of error.
  Make them `estimate(...)` objects with a tolerance, and put a `checkpoint(...)` where a wrong estimate
  would make the next step pointless (e.g. after hovering over an object, before descending to grasp).
- **The loop:**
  1. `begin_episode(instruction)`.
  2. Observe; estimate the objects (positions and sizes).
  3. Write the program; `run_program(code, title, scene=[{name, xy, size}, ...])` with the objects you
     believe in. **Your tools' instructions say whether this session previews programs in a simulated
     twin.** Usually it does not: programs run directly on the live robot, so check reach with
     `check_reach` first and keep first motions slow. When it does, a failed preview comes back with the
     reason - fix and resubmit. Trust the preview on reach, the plan and safety; not on grasps - its
     grasp physics often differ from the real robot, and it places objects square (no rotation).
  4. Then it waits for a **human to approve it** on the operator page: tell the user it is waiting.
  5. Follow it with `program_status(id, wait_s=30)`. At a checkpoint you get the evidence images:
     `resume_program(id, "continue" | "refine" | "abort", updates={"name": [x, y]})`. A refinement must
     stay within the estimate's tolerance; a bigger correction means revising the program. While a program
     waits at a checkpoint, keep calling `program_status` (e.g. while you ask the user something): after
     10 minutes without any call from you it is aborted, on the assumption that you have gone.
  6. If a step fails, `get_program_trace` (and `get_keyframe`) tell you what happened. Fix the cause, not
     the symptom; do not retry the same thing unchanged.
  7. Verify the outcome with the cameras, then `end_episode(outcome, notes)`. In `notes`, write what you
     learned (what worked, what failed and why) - it is kept as data.
- **When something looks wrong**, call `stop()` and tell the user.

## Skills (your own, reusable)

A skill is a procedure you can hand to your next session: it ran on the real robot, cleanly, as written. Getting a
task done for the first time is **discovery**, not yet a skill. Keep what you learn as a **draft**, then validate it.

### Files

```
.agents/skills/<name>/SKILL.md      frontmatter: name, description, status; then what it does, its parameters,
                                    preconditions and end state, what you learned (failure symptoms and fixes)
.agents/skills/<name>/skill.py      the code (programs cannot import it: copy it into the program)
.agents/skills/<name>/history.jsonl one line per use or change: {"date", "episode", "program", "event", "outcome", "notes"}
.agents/skills/<name>/revision/     a proposed change to a validated skill, while it is tested: REVISION.md (the
                                    changed SKILL.md text, with `revision_status` in its frontmatter) and revision.py
```

`status` is `draft` or `validated`. Start the `description` of a draft with `[draft]`, so the status shows in the list
of skills. Update the history every time you use or change a skill. Older history lines may have no `event`.
If your file tool can only write whole files, read the file in full first and write all of it back, so no line is lost.

### A clean run

A run of the skill **as written** that:
- succeeds in one program;
- has no failed or refused step, no abort and no STOP;
- has refinements only within the estimates' tolerances, at the checkpoints the skill itself defines;
- needs no code change from start to finish.

A refine loop the skill defines (e.g. "refine up to 3 times") is fine. A STOP, by you or by the user, makes a run not
clean. Any run that is not clean resets the count of clean runs in a row.

A success after aborts, code edits between programs, or recovery programs is discovery. Write the draft from what you
learned, and say in the history what went wrong.

### Draft -> validated

1. **Save a draft whenever you have learned something worth keeping**: after a discovery episode, before a long
   session runs out of context, or when the user ends the session. A draft says what is known, what is guessed, and
   what has not been tried yet.
2. **Iterate.** Use the draft in the next episodes. Change the code or method when a run shows a cause it should
   handle, and record every change in the history (`"event": "changed"`).
3. **Get repeats.** A draft needs **2 clean runs in a row, with the code unchanged in between**, on different layouts.
   If a draft claims to work for both arms, one of the runs must use the other arm. Once the task is done, tell the
   user how many clean runs the draft has, and ask for a repeat episode (see "Test layouts" below).
4. **Ask the user to approve it.** Show the clean runs (episodes and programs), what the skill covers and what it
   does not. Only after the user says yes: set `status: validated`, remove `[draft]` from the description, and add a
   history line `"event": "validated"` that records the user's approval.

Never mark a skill validated without the user's explicit approval in this session.

### Refining a validated skill

A validated skill stays as it is while you improve it, so it keeps working for other tasks:
1. When a run shows something the skill should handle better, or it fails for a reason its code should have handled,
   record it in the history (`"event": "failed"` or `"issue"`). Then write the change in `revision/` (`REVISION.md`
   with `revision_status: testing`, and `revision.py`), not in the skill itself. Do not name it `SKILL.md`: it would
   be listed as a second skill.
2. Test the revision **live**, by the same bar as a draft: 2 clean runs in a row, on different layouts, with the
   revision unchanged in between. Use the validated version for everything else meanwhile. Log each run of the
   revision in the skill's history with `"event": "revision run"`.
3. Ask the user to approve it, showing the runs and what changed. Only after the user says yes: copy `revision.py` into
   `skill.py` and the revised text into `SKILL.md`, keeping `SKILL.md`'s own frontmatter (`status: validated`); set `revision_status: merged` in `REVISION.md`, and add a history line
   `"event": "revised"`. If the user declines, or the revision does not work out, set `revision_status: dropped` and
   record why. You cannot delete files: a merged or dropped revision stays until the next revision overwrites it.

Fixing wording or history lines needs no runs.

### Test layouts

Repeats need the objects in new places. Either ask the user to set up a layout (say where, roughly), or set it up
yourself with the robot: move the objects with **validated** skills, in a program of its own titled "test layout"
(in the current episode), and tell the user what you are doing first. Never test a draft or a revision by setting up its own layout with it. If no validated skill can
put the objects where you need them, ask the user.

### One skill, one capability

- A skill does **one thing you would reuse in a different task**. It has a clear start state (preconditions) and end
  state, for example "grasp a small block off the table" or "put a held object into a container". It takes the arm, positions and sizes as parameters.
- Split when a part is useful on its own, or has its own checkpoints and failure modes. Do not split a few lines that
  only make sense together.
- A larger task gets a **composite** skill, which lists the skills it uses (`uses: [a, b]` in the frontmatter) and
  adds only the sequencing between them and their hand-over states (what one leaves, what the next needs). Copy the
  parts' code into the program, and say which version of each part you copied: the date and episode of its last
  `changed`, `validated` or `revised` history line.
- A composite is validated by its own clean runs, and only once all its parts are validated.
- Before writing a new skill, check whether an existing one already covers it, or should be split. To split a
  validated skill, write the parts as new drafts (and the composite that replaces it); the old skill stays validated
  and in use until they are validated, and then the user decides whether to retire it.

## Read these two documents before you act

They are part of this brief. Read both in full at the start of every session, before writing any program:

1. **`../stack/rig/embodiment.md`** (from this folder; `stack/rig/embodiment.md` in the repository) - the robot:
   the rig and its coordinates, each arm's joints and directions, the gripper, jaws and wrist camera at each
   wrist roll, and the cameras.
2. **`../lib/robot_use/README.md`** (`lib/robot_use/README.md` in the repository) - the library your programs
   use: every call, what it does and returns, conditions, checkpoints and estimates, the runner's limits.

If your agent shows their contents below already (Claude Code loads the two `@` lines), you have read them.
Otherwise open the two files with your file-reading tool now.

## The robot

@../stack/rig/embodiment.md

## The library

@../lib/robot_use/README.md

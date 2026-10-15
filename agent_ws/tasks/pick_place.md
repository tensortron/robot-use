# Task: pick and place

The user will name an object and a destination, for example: "pick up the red block and put it in the
yellow cup".

Do it with the robot, following the loop in AGENTS.md. Choose the arm that can reach both the object and
the destination (or plan a handover if neither can).

**Success** means the object ends inside / on the destination, the gripper is open and clear of it, and
the arm is back at rest or holding clear of the workspace. Check it on the ego camera before you call it
a success.

**Report** at the end: the outcome, how many programs you ran, what failed along the way and why, and
anything you would do differently. Save or update your skills as AGENTS.md ("Skills") says: a draft after
discovery; tell the user how many clean runs it has, and ask for a repeat or for approval.

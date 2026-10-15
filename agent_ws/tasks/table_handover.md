# Task: table hand-off between the two arms

The user will name an object and a destination, for example: "move the blue block into the pink cup,
using both arms".

Do it with the robot, following the loop in AGENTS.md, and use **both** arms: the arm nearer the object
picks it up and sets it down on the table somewhere the other arm can reach; that arm then picks it up
and puts it in / on the destination. The first arm should be clear of the object before the second arm
reaches for it.

**Success** means the object ends inside / on the destination, both grippers are open and clear of it,
and both arms are back at rest or holding clear of the workspace. Check it on the ego camera before you
call it a success.

**Report** at the end: the outcome, how many programs you ran, what failed along the way and why, and
anything you would do differently. Save or update your skills as AGENTS.md ("Skills") says: a draft after
discovery; tell the user how many clean runs it has, and ask for a repeat or for approval.

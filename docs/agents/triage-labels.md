# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

_Edit the "Label in our tracker" column to match whatever vocabulary you actually use._

## One deliberate exception: spec issues carry no category role

A **spec** issue — the umbrella an epic-style build is decomposed into tickets from — carries a
*state* role but no `bug`/`enhancement` category. In this tracker that is **#1**, the multi-face
spec. It is neither broken nor a single feature; it is the contract the tickets under it are written
against, and forcing a category onto it makes the tracker claim something false about it.

So when triaging here: one state role for every issue, one category role for every issue **except**
a spec umbrella. Sub-issue children of a spec do carry both, because each child is a concrete change.

_All five labels exist on `EmberLightVFX/ComfyUI-H3-FaceRefine`: `wontfix` was already a
default GitHub label; `needs-triage`, `needs-info`, `ready-for-agent` and `ready-for-human`
were created during setup._

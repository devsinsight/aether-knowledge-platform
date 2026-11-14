# Work records

Work record files written by agents, one per user story. The publisher reads them **by path** from this folder (mounted read-only into the `publisher` service) and publishes them to the wiki under `Work Records / Work Records: <feature> / Work Record: <story>`. The wiki page is the record; these files are inputs.

Template (AIEF `templates/work-record.md`). Every field marked required must be filled; the publisher never fills one.

```markdown
---
key: work-record-<story-key>          # required; derived from the story key
type: work-record                     # required
story: <story key and work item ID>   # required, for example story-scoring-chain-window (ADO 123)
story_revision: <work item revision>  # required; the acceptance criteria are cited at this revision
feature: <feature key>                # required
state: draft                          # required; draft until a person validates (for example, on merge)
author: <agent or person>             # required
owner: <person who validates>
pull_request: <URL>                   # once it exists
---

<!-- Work record: the analysis and resolution of one user story (AIEF ADR-006). One page per story, under Work Records / <feature> / <story>. -->
<!-- The publisher reads this front matter, rejects the file if a required field is missing, and writes it into the Page Properties macro. It never fills a field. -->

# <story title>: work record

## Analysis

How the story was understood: the rules, decisions and gaps consulted (by key and version), the assumptions made, and the questions raised as Knowledge Gaps.

## Approach

The options considered, the one chosen and why. A significant architectural choice gets its own ADR; link it here.

## Acceptance criteria resolution

The criteria are cited by ID at the story revision above, not copied.

| AC | How it was resolved | Tests that verify it | Commits |
| --- | --- | --- | --- |
| AC-1 | | | |

## Implementation

What changed, by component, with links to the pull request and commits. Code is linked, never pasted.

## Open items

Gaps, follow-ups or limits found during the work, each as a Knowledge Gap with an owner.

## Validation

Who validated this record and when (filled by the publisher on `validate`).
```

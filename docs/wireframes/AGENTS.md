These are references for the screens. Build only the parts which the current api supports already. Dont introduce any new feature that will required backend re working.

Detailed design tokens are in DESIGN.md

New UI flow:

- main nav is on the left instead of top: dashboard, datasets, models, jobs
- dashboard is the new home page: shows current active jobs, recently finished jobs, and a glimpse of the calibration data.
- dataset manager: allows to upload/import datasets in jsonl. shows previously used datasets as well.
- dataset detail view: shows what the current dataset validation page shows
- jobs: shows list of all previous training jobs
- initiate new job: this will have its own sub steps - each step can be displayed in the top/left nav
  - select dataset and base model
  - tune hyperparams
  - review and launch
  - job in progress screen/ terminal job screen
- model explorer: users can browse base model catalog + import and browse models from huggingface, to use a base model, user will have to 'save' it - this will actually download the model

Only run the frontend checks for UI rework - dont run the full test suite.

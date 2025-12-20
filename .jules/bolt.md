
## 2024-05-23 - [Missing Filter Context in Background Workers]
**Learning:** Found a critical performance issue where a background worker (DatabaseWorker) was loading ALL video clips from the database because the context (project_id) was not passed from the UI widget.
**Action:** Always ensure context identifiers (like project_id) are explicitly passed to background workers and used in database queries to utilize indexes and reduce data volume. Checked for N+1 issues and realized simple eager loading is not enough if the base query is too broad.

# Drone Processing pipeline 

### Author: Alejandro

### Databricks Runtime

### Set up Steps

This repo contains the scripts/notebooks used to run a set of Databricks Jobs & Pipelines. This README explains how to recreate each job in the Databricks UI. We'll document them one at a time, starting with **Job 1: TABLE_UPDATE**.

## Job 1: `TABLE_UPDATE`

This job updates the flight and orthomosaic tables and is made up of **3 sequential tasks**, each running on **Serverless** compute.

### Pipeline flow

```
1_Flights_update  →  2_update_tif  →  3_orquestator
```

### Steps to create this job in Databricks

1. Go to **Jobs & Pipelines** in the Databricks workspace sidebar.
2. Click **Create Job / Pipeline** and name it `TABLE_UPDATE`.

3. **Add Task 1 — `1_Flights_update`**
   - Type: Notebook
   - Source: point to the `drone_flight_table_generation` notebook in this repo
   - Compute: **Serverless**
   - Inside this notebook, you need to update the `catalog` name (the space where the table will be created), the `schema` name, and the `root_path` (where all your flight data is stored). Every line marked with `#change` must be updated to match your own environment.

4. **Add Task 2 — `2_update_tif`**
   - Type: Notebook (or Python script)
   - Source: point to the `drone_orthomosaic_table_generation` notebook in this repo
   - Compute: **Serverless**
   - Set **Depends on**: `1_Flights_update`
   - Inside this notebook, you need to update the `catalog` name (the space where the table will be created), the `schema` name. Every line marked with `#change` must be updated to match your own environment.

5. **Add Task 3 — `3_orquestator`**
   - Type: Notebook (or Python script)
   - Source: point to the `orthomosaic_orquestator` notebook in this repo
   - Compute: **Serverless**
   - Set **Depends on**: `2_update_tif`
   - Inside this notebook, you need to update the `catalog` name (the space where the table will be created) and the `schema` name. Additionally, in the final cell of the notebook, you must update the ID of the job to be triggered — this refers to Job 2, so you'll need to enter the ID that Databricks generates for it. Every line marked with `#change` must be updated to match your own environment.

6. Save the job. The tasks will run in order: `1_Flights_update → 2_update_tif → 3_orquestator`.

7. Run the job manually once to confirm it completes successfully before scheduling it, or setting up a trigger.

---

---

## Job 2: `PIPELINE_ORTHOMOSAIC`

This job runs the Agisoft Metashape orthomosaic processing and then deactivates the Agisoft license once finished. It has **2 tasks**, both running on **Agisoft Metashape** compute.

### Pipeline flow

```
4_Agisoft_Processing  →  (if all done)  →  deactivate_agisoft_license
```

### Steps to create this job in Databricks

1. Go to **Jobs & Pipelines** and click **Create Job**. Name it `PIPELINE_ORTHOMOSAIC`.
2. **Add Task 1 — `4_Agisoft_Processing`**
   - Source: point to the `agisoft_drone_pipeline_rgb` notebook/script in this repo
   - Compute: **Agisoft Metashape** cluster
   - In the final cell of the notebook, you must update the ID of the job to be triggered — this refers to Job 3, so you'll need to enter the ID that Databricks generates for it. Every line marked with `#change` must be updated to match your own environment.

3. **Add Task 2 — `deactivate_agisoft_license`**
   - Source: point to the `deactivate_agisoft_license` notebook/script in this repo
   - Compute: **Agisoft Metashape** cluster
   - Set **Depends on**: `4_Agisoft_Processing`, with the run condition set to **"All done"** (runs whether the previous task succeeded or failed, to ensure the license is always released)
4. Save the job and run it once manually to confirm both steps complete and the license is released correctly.

---

## Job 3: `PLOT_CLIPS_GENERATION`

This job generates the plot clip tables and conditionally triggers clip generation. It has **6 tasks**, all running on **Serverless** compute, including a conditional branch.

### Pipeline flow

```
1_Flight_table → 2_clip_table_gen → 2_5_Update_ortho_table → 3_orquestator → check_pending_clips → (True) → pending_clips_gen
```

### Steps to create this job in Databricks

1. Go to **Jobs & Pipelines** and click **Create Job**. Name it `PLOT_CLIPS_GENERATION`.
2. **Add Task 1 — `1_Flight_table`**
   - Source: `drone_flight_table_generation` notebook
   - Compute: **Serverless**
   
3. **Add Task 2 — `2_clip_table_gen`**
   - Source: `drone_plot_clipped_table` notebook
   - Compute: **Serverless**
   - Depends on: `1_Flight_table`
   - Inside this notebook, you need to update the `catalog` name (the space where the table will be created), the `schema` name. Every line marked with `#change` must be updated to match your own environment.
4. **Add Task 3 — `2_5_Update_ortho_table`**
   - Source: `drone_orthomosaic_table_generation` notebook
   - Compute: **Serverless**
   - Depends on: `2_clip_table_gen`
   
5. **Add Task 4 — `3_orquestator`**
   - Source: `plot_clip_orchestrator` notebook
   - Compute: **Serverless**
   - Depends on: `2_5_Update_ortho_table`
   - Inside this notebook, you need to update the `catalog` name (the space where the table will be created), the `schema` name. Every line marked with `#change` must be updated to match your own environment.
6. **Add Task 5 — `check_pending_clips`** (Condition task)
   - Type: **If/else condition**
   - Condition: `{{proceed}} == "true"` (this references a task value/parameter named `proceed` set earlier in the pipeline)
   - Depends on: `3_orquestator`
7. **Add Task 6 — `pending_clips_gen`**
   - Source: `Plot_clipping` notebook (path: `Shared/pipeline_drones/Plot_clipping`)
   - Compute: **Serverless**
   - Depends on: `check_pending_clips` → **True** branch only
   - In the final cell of the notebook, you must update the ID of the job to be triggered — this refers to Job 4, so you'll need to enter the ID that Databricks generates for it. Every line marked with `#change` must be updated to match your own environment.
8. Save the job and run it once manually to confirm the conditional branch behaves as expected.

---

## Job 4: `PHENO_I`

This is a single-task job that runs the PhenoI processing pipeline on a custom compute cluster.


### Steps to create this job in Databricks

1. Go to **Jobs & Pipelines** and click **Create Job**. Name it `PHENO_I`.
2. **Add Task 1 — `PHENO_1_RUNNER`**
   - Source: point to the `PhenoI_runner` notebook/script in this repo
   - Compute: **Pheno-I Custom Compute** (a dedicated custom cluster — make sure it's created/configured in your workspace before assigning it here)
   - Inside this notebook, you need to update the `catalog` name (the space where the table will be created), the `schema` name. Every line marked with `#change` must be updated to match your own environment.
3. Save the job and run it once manually to confirm it completes successfully.

---

## Summary of all jobs

| Job | Tasks | Compute type(s) |
|-----|-------|------------------|
| `TABLE_UPDATE` | 3 (sequential) | Serverless |
| `PIPELINE_ORTHOMOSAIC` | 2 | Agisoft Metashape |
| `PLOT_CLIPS_GENERATION` | 6 (with conditional branch) | Serverless |
| `PHENO_I` | 1 | Pheno-I Custom Compute |
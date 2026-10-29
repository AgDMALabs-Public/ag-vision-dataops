# Hiphen Data Pull
* Author: Dan Williams
* Date: 7/1/2026

# Parameters
* Set the Parameters at the top of the notebook for your workspace.
* The Parameters will need to match between the two notebooks.

##Directions:
* Get an API key from Hipen. Then use the notebook called set_api_key.ipynb to set the key.
* Run generate_hiphen_summary.ipynb notebook. This will generate the table with the hiphen summary metrics. This table is used to tell the pull code which missions need to be downloaded and which ones are downloaded.
* Run the pull_hiphen_data.ipynb. This will pull the plot images and ortho images and save them to the right dir for use.
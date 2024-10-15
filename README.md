# Basic Workflow

### First, the system starts the edge server and loads the weight file of the edge-side model at the same time.
### Secondly, the edge captures image data at the predetermined time interval and then uploads that data to OBS (Object Storage Service).
### The complex model in the cloud annotates the image data and retrains the edge-side model using ModelArts. Additionally, the fine-tuned edge-side model is also stored in OBS.
### The edge updates the optimal model weight file (best.pt) from the cloud.
### Finally, the edge performs the inference detection task for the next stage.
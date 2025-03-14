

# Serving machine learning models on edge devices

In this tutorial, we will benchmark machine learning models on a low-resource edge device (a Raspberry Pi 5 with Arm Cortext A76 processor). We will measure the inference time of:

* a baseline model
* a model with INT8 quantization
* and the same models, using the [Arm Compute Library](https://github.com/ARM-software/ComputeLibrary) execution provider

To run this experiment, you should have already created an account on Chameleon, and become part of a project. 




## Context

The premise of this example is as follows: You are working as a machine learning engineer at a small startup company called GourmetGram. They are developing an online photo sharing community focused on food. You have developed a convolutional neural network in Pytorch that automatically classifies photos of food into one of a set of categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit.

Now that you have trained a model, you are preparing to serve predictions using this model. Your manager has advised that since GourmetGram is an early-stage startup, they can't afford much compute for serving models. Your manager wants you to prepare a few different options, that they will then price out among cloud providers and decide which to use:

* inference on a server-grade CPU (AMD EPYC 7763). Your manager wants to see an option that has less than 3ms median inference latency for a single input sample, and has a batch throughput of at least 1000 frames per second.
* inference on a server-grade GPU (A100). Since GourmetGram won't be able to afford to load balance across several GPUs, your manager said that the GPU option must have strong enough performance to handle the workload with a single GPU node: they are looking for less than 1ms median inference latency for a single input sample, and a batch throughput of at least 5000 frames per second.
* inference on end-user devices, as part of an app. For this option, the model itself should be less than 5MB on disk, because users are sensitive to storage space on mobile devices. Because the total prediction timme will not include any network delay when the model is on the end-user device, the "budget" for inference time is larger: your manager wants less than 15ms median inference latency for a single input sample on a low-resource edge device (ARM Cortex A76 processor).

You have [evaluated your model on server-grade CPU and GPU already](https://teaching-on-testbeds.github.io/serve-model-chi/); now you are ready to benchmark on a low-resource edge device.



## Experiment resources 

For this experiment, we will provision one Raspberry Pi 5 at CHI@Edge. Edge devices, like bare metal devices, need to be reserved in advance.



## Create a lease for a GPU server



For this experiment, we will reserve a 2-hour block on a Raspberry Pi 5.

We can use the OpenStack graphical user interface, Horizon, to submit a lease. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/)
* click "Experiment" > "CHI@Edge"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.



Then, 

* On the left side, click on "Reservations" > "Leases", and then click on "Device Calendar". In the "Vendor" drop down menu, change the type to "Raspberry Pi" to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC. You can use [WolframAlpha](https://www.wolframalpha.com/) or equivalent to convert to your local time zone. 
* Once you have identified an available two-hour block in UTC time that works for you in your local time zone, make a note of:
  * the start and end time of the time you will try to reserve. (Note that if you mouse over an existing reservation, a pop up will show you the exact start and end time of that reservation.)
* Then, on the left side, click on "Reservations" > "Leases", and then click on "Create Lease":
  * set the "Name" to <code>serve_edge_<b>netID</b></code> where in place of <code><b>netID</b></code> you substitute your actual net ID.
  * set the start date and time in UTC. To make scheduling smoother, please start your lease on an hour boundary, e.g. `XX:00`.
  * modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time to two hours as directed. Also, to avoid a potential race condition that occurs when one lease starts immediately after another lease ends, you should end your lease five minutes before the end of an hour, e.g. at `YY:55`.
  * Click "Next".
* Click "Next". (We won't include any network resources in this lease.)

* On the "Devices" tab, 
  * check the "Reserve devices" box
  * leave the "Minimum number of hosts" and "Maximum number of hosts" at 1
  * in "Resource properties", specify `machine_name` as `raspberrypi5`. Or, to reserve a specific device, specify its `uid`. These are the UUIDs of our Pi 5s:


| Raspberry Pi Node | UUID                                    |
|-------------------|-----------------------------------------|
| `nyu-rpi5-01`     | `c516acb2-4c88-42be-857f-2f9eb4139f99`  |
| `nyu-rpi5-02`     | `8334d598-d25d-4dbb-a416-d90f8e93ccc4`  |
| `nyu-rpi5-03`     | `a755b236-580e-4040-874c-80501f00f954`  |
| `nyu-rpi5-04`     | `9a7823ea-bf40-4141-b7d7-943bfb389091`  |
| `nyu-rpi5-05`     | `54a6e248-cbb2-472d-bde7-0b4ac3bd911a`  |
| `nyu-rpi5-06`     | `52341ad4-ff91-4516-a3ec-88687a8d984b`  |
| `nyu-rpi5-07`     | `7bfe3fe8-6f1e-41a4-8d9a-aa9668db8805`  |

* Then, click "Create".

Your lease status should show as "Pending". Click on the lease to see an overview. It will show the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease. Make sure that the lease details are correct.




## At the beginning of your edge device lease


At the beginning of your edge device lease time, you will continue with the next step, in which you will launch a container on the device! To begin this step, open this experiment on Trovi:

* Use this link: [Serving machine learning models on edge devices](https://chameleoncloud.org/experiment/share/a1662022-9017-45b1-9b96-31705ca20358) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it, including the notebok to launch the container.







## Launch a container on an edge device - with python-chi

At the beginning of the lease time for your device, we will use the `python-chi` Python API to Chameleon to launch a container on it, using OpenStack's Zun container service. 

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected. Make sure the site is set to CHI@Edge.


```python
from chi import container, context, lease
import os
import chi

context.version = "1.0" 
context.choose_project()
context.choose_site(default="CHI@Edge")
```


Change the string in the following cell to reflect the name of *your* lease (**with your own net ID**), then run it to get your lease:


```python
l = lease.get_lease(f"serve_edge_netID") 
l.show()
```


The status should show as "ACTIVE" now that we are past the lease start time.



We will use the lease to launch a Jupyter notebook container on a Raspberry Pi 5 edge device. 

> **Note**: the following cell brings up a container only if you don't already have one with the same name! (Regardless of its error state.) If you have a container in ERROR state already, delete it first in the Horizon GUI before you run this cell.



```python
username = os.getenv('USER') # exp resources will have this suffix
c = container.Container(
    name = f"node-serve-edge-{username}".replace('_', '-'),
    reservation_id = l.device_reservations[0]["id"],
    image_ref = "quay.io/jupyter/minimal-notebook:latest", 
    exposed_ports = [8888]
)
c.submit(idempotent=True)
```



Then, we'll associate a floating IP with the container, so that we can access the Jupyter service running in it.


```python
c.associate_floating_ip()
```


In the output above, make a note of the floating IP that has been assigned to your container.


Let's retrieve a copy of these materials on the container:




```python
stdout, code = c.execute("git clone https://github.com/teaching-on-testbeds/serve-edge-chi.git")
print(stdout)
```




```python
stdout, code = c.execute("mv serve-edge-chi/workspace/models work/")
print(stdout)
```



```python
stdout, code = c.execute("mv serve-edge-chi/workspace/measure_pi.ipynb work/")
print(stdout)
```



and, install the ONNX runtime Python module:


```python
stdout, code = c.execute("python3 -m pip install onnxruntime")
print(stdout)
```



Finally, we will get the container logs. Run:



print(chi.container.get_logs(c.id))




and look for a line like

```
    http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```


Paste this into a browser tab, but in place of 127.0.0.1, substitute the floating IP assigned to your container, to open the Jupyter notebook interface that is running *on your Raspberry Pi 5*.

Then, in the file browser on the left side, open the “work” directory and find the `measure_pi.ipynb` notebook to continue.




## Measure inference performance of ONNX model on low-resource edge device 

Now, we're going to benchmark a couple of previously created ONNX models on our low-resource edge device.

You will execute this notebook *in a Jupyter container running on an edge device*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.



```python
import os, time
import numpy as np
import onnxruntime as ort
```



We'll define a benchmark function. For convenience (since we don't need real data for benchmarking) we will use random "fake" samples to evaluate our models' inference performance.


```python
def benchmark_session(ort_session):

    ## Benchmark inference latency for single sample

    num_trials = 100  # Number of trials
    input_shape = ort_session.get_inputs()[0].shape  # Get expected input shape
    input_dtype = np.float32  # Adjust dtype as needed
    fixed_shape = (1, *input_shape[1:])  

    # Generate a single dummy sample with random values
    single_sample = np.random.rand(*fixed_shape).astype(input_dtype)

    # Warm-up run
    ort_session.run(None, {ort_session.get_inputs()[0].name: single_sample})

    latencies = []
    for _ in range(num_trials):
        start_time = time.time()
        _ = ort_session.run(None, {ort_session.get_inputs()[0].name: single_sample})
        latencies.append(time.time() - start_time)

    print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
    print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
```


Now, let's evaluate our "baseline" ONNX model:


```python
onnx_model_path = "models/food11.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```


the model quantized with dynamic quantization:


```python
onnx_model_path = "models/food11_quantized_dynamic.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```


and the model quantized with static quantization, for which we permit up to 0.05 decrease in accuracy:


```python
onnx_model_path = "models/food11_quantized_aggressive.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```



When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)




<hr>

<small>Questions about this material? Contact Fraida Fund</small>

<hr>

<small>This material is based upon work supported by the National Science Foundation under Grant No. 2230079.</small>

<small>Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.</small>
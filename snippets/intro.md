
::: {.cell .markdown}

# Serving machine learning models on edge devices

In this tutorial, we will benchmark machine learning models on a low-resource edge device (a Raspberry Pi 5 with Arm Cortext A76 processor). We will measure the inference time of:

* a baseline model
* a model with INT8 quantization
* and the same models, using the [Arm Compute Library](https://github.com/ARM-software/ComputeLibrary) execution provider

To run this experiment, you should have already created an account on Chameleon, and become part of a project. 

:::


::: {.cell .markdown}

## Context

The premise of this example is as follows: You are working as a machine learning engineer at a small startup company called GourmetGram. They are developing an online photo sharing community focused on food. You have developed a convolutional neural network in Pytorch that automatically classifies photos of food into one of a set of categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit.

Now that you have trained a model, you are preparing to serve predictions using this model. Your manager has advised that since GourmetGram is an early-stage startup, they can't afford much compute for serving models. Your manager wants you to prepare a few different options, that they will then price out among cloud providers and decide which to use:

* inference on a server-grade CPU (AMD EPYC 7763). Your manager wants to see an option that has less than 3ms median inference latency for a single input sample, and has a batch throughput of at least 1000 frames per second.
* inference on a server-grade GPU (A100). Since GourmetGram won't be able to afford to load balance across several GPUs, your manager said that the GPU option must have strong enough performance to handle the workload with a single GPU node: they are looking for less than 1ms median inference latency for a single input sample, and a batch throughput of at least 5000 frames per second.
* inference on end-user devices, as part of an app. For this option, the model itself should be less than 5MB on disk, because users are sensitive to storage space on mobile devices. Because the total prediction timme will not include any network delay when the model is on the end-user device, the "budget" for inference time is larger: your manager wants less than 15ms median inference latency for a single input sample on a low-resource edge device (ARM Cortex A76 processor).

You have [evaluated your model on server-grade CPU and GPU already](https://teaching-on-testbeds.github.io/serve-model-chi/); now you are ready to benchmark on a low-resource edge device.

:::

::: {.cell .markdown}

## Experiment resources 

For this experiment, we will provision one Raspberry Pi 5 at CHI@Edge. Edge devices, like bare metal devices, need to be reserved in advance.

:::

::: {.cell .markdown}

## Create a lease for an edge device

:::

::: {.cell .markdown}

For this experiment, we will reserve a 2-hour block on a Raspberry Pi 5.

We can use the OpenStack graphical user interface, Horizon, to submit a lease. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/)
* click "Experiment" > "CHI@Edge"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.

:::

::: {.cell .markdown}

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

:::


::: {.cell .markdown}

## At the beginning of your edge device lease


At the beginning of your edge device lease time, you will continue with the next step, in which you will launch a container on the device! To begin this step, open this experiment on Trovi:

* Use this link: [Serving machine learning models on edge devices](https://chameleoncloud.org/experiment/share/a1662022-9017-45b1-9b96-31705ca20358) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it, including the notebok to launch the container.


:::


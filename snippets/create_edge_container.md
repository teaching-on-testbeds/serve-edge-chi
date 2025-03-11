


::: {.cell .markdown}

## Launch a container on an edge device - with python-chi

At the beginning of the lease time for your device, we will use the `python-chi` Python API to Chameleon to launch a container on it, using OpenStack's Zun container service. 

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected. Make sure the site is set to CHI@Edge.

:::

::: {.cell .code}
```python
from chi import container, context, lease
import os

context.version = "1.0" 
context.choose_project()
context.choose_site(default="CHI@Edge")
```
:::

::: {.cell .markdown}

Change the string in the following cell to reflect the name of *your* lease (**with your own net ID**), then run it to get your lease:

:::

::: {.cell .code}
```python
l = lease.get_lease(f"serve_edge_netID") 
l.show()
```
:::

::: {.cell .markdown}

The status should show as "ACTIVE" now that we are past the lease start time.

:::

::: {.cell .markdown}

We will use the lease to launch a Jupyter notebook container on a Raspberry Pi 5 edge device. 

> **Note**: the following cell brings up a container only if you don't already have one with the same name! (Regardless of its error state.) If you have a container in ERROR state already, delete it first in the Horizon GUI before you run this cell.

:::


::: {.cell .code}
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
:::


::: {.cell .markdown}

Then, we'll associate a floating IP with the container, so that we can access the Jupyter service running in it.

:::

::: {.cell .code}
```python
c.associate_floating_ip()
```
:::

::: {.cell .markdown}

In the output above, make a note of the floating IP that has been assigned to your container.
:::

::: {.cell .markdown}

Let's retrieve a copy of these materials on the container:


:::


::: {.cell .code}
```python
stdout, code = c.execute("git clone https://github.com/teaching-on-testbeds/serve-edge-chi.git")
print(stdout)
```
:::




::: {.cell .code}
```python
stdout, code = c.execute("mv serve-edge-chi/workspace/models work/")
print(stdout)
```
:::



::: {.cell .code}
```python
stdout, code = c.execute("mv serve-edge-chi/workspace/measure_pi.ipynb work/")
print(stdout)
```
:::


::: {.cell .markdown}

and, install the ONNX runtime Python module:

:::

::: {.cell .code}
```python
stdout, code = c.execute("python3 -m pip install onnxruntime")
print(stdout)
```
:::


::: {.cell .markdown}

Finally, we will get the container logs. Run:

:::


print(chi.container.get_logs(c.id))


::: {.cell .markdown}


and look for a line like

```
    http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```


Paste this into a browser tab, but in place of 127.0.0.1, substitute the floating IP assigned to your container, to open the Jupyter notebook interface that is running *on your Raspberry Pi 5*.

Then, in the file browser on the left side, open the “work” directory and find the `measure_pi.ipynb` notebook to continue.

:::

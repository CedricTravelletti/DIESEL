""" Define the various types of computing clusters that can 
be used to run the computation.

"""
from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster


def UbelixCluster(n_nodes, mem_per_node=16, cores_per_node=1, 
        partition="epyc2", qos="job_epyc2"):
    """ Provision a Daks cluster on the Ubelix cluster of UniBern.

    Parameters
    ----------
    n_nodes: int
    mem_per_node: int, default=16
        Memory per node in GB.
    cores_per_node: int, default=1
    partition: string
        Under which queue to submit the job.
    qos: string
        QOS queue under which to submit the job.

    Returns
    -------
    cluster

    """
    mem_per_node = "{} GB".format(mem_per_node)
    cluster = SLURMCluster(
        cores=cores_per_node,
        memory=mem_per_node,
        death_timeout=6000,
        walltime="06:00:00",
        job_extra=['--qos="{}"'.format(qos), '--partition="{}"'.format(partition)]
    )

    # Manually define the size of the cluster.
    cluster.scale(n_nodes)
    return(cluster)

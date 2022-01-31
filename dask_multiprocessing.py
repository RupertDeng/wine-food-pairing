import dask.dataframe as ddf

def dask_compute(df, npartitions, nworkers, func, *args):
  """
  Utilize dask framework to achieve multi-processing on large data set.

  Args:
  - df: original pandas dataframe
  - npartitions: number of partitions dask will split from the original df by row
  - nworkers: number of processes to compute on the partitions
  - func: actual process function on the data, takes in dask df and args, and returns a dataframe after processing the column

  Returns a single processed pandas dataframe

  workflow: dask will split the original dataframe into multiple partitions, and compute them with multiple processes in parallel, then combine them back into one dataframe.
  
  It is advised to keep the column dtype the same as original df.
  If different, need to specify the outcome dtype by the meta arg in map_partitions. Refer to dask documentation.
  """

  return ddf.from_pandas(df, npartitions=npartitions).map_partitions(func, *args, meta=df, align_dataframes=False).compute(scheduler='processes', num_workers=nworkers)

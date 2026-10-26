# OutOfMemoryError: Java Heap Space and Python Memory Exhaustion
Category: Runtime Memory Management & Performance
Technologies: Java, Python, JVM, Linux, Kubernetes

When processing large datasets or high-concurrency requests:
`java.lang.OutOfMemoryError: Java heap space` or `MemoryError` or process terminated by OS `Killed (exit code 137)`.

## Root Cause
1. In Java/JVM applications, the maximum heap allocation limit (`-Xmx`) default is insufficient for the memory consumption of active objects.
2. In Python, an un-chunked file read (e.g. `file.read()` or unpaginated SQL query) loaded gigabytes of data into RAM at once.
3. The Linux Kernel Out-Of-Memory (OOM) killer terminated the process to protect system stability.

## Resolution Steps
1. In Java applications, increase the maximum heap size via JVM arguments:
   `-Xms512m -Xmx4g`
2. In Python, process data using streams or generators:
   Use `chunksize` in Pandas (`pd.read_csv('large.csv', chunksize=10000)`).
3. In containerized environments, verify container memory limits in your Compose or Kubernetes spec:
   `resources.limits.memory: 4Gi`

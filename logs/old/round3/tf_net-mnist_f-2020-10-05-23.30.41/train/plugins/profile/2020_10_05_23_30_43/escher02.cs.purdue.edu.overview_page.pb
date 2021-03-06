�	Z��լ��?Z��լ��?!Z��լ��?	B�'�Q@B�'�Q@!B�'�Q@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Z��լ��?,�F<�Ͱ?AT9�)9'�?Y�`6��?*	V-�}F@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��;��J�?!�(ax�C@)�[t��z�?1�F��A@:Preprocessing2F
Iterator::Model5��o�h�?!������A@)��̰Q�?1}��d;:8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�%�2��?!}��|��2@)��OVw?1���e�)@:Preprocessing2U
Iterator::Model::ParallelMapV2*��% �t?!�|D-��&@)*��% �t?1�|D-��&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���{*��?!�0�P@)�����f?1���]~�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��+H3f?!� �&9@)��+H3f?1� �&9@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�^Pjd?!˞� �(@)�^Pjd?1˞� �(@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9B�'�Q@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	,�F<�Ͱ?,�F<�Ͱ?!,�F<�Ͱ?      ��!       "      ��!       *      ��!       2	T9�)9'�?T9�)9'�?!T9�)9'�?:      ��!       B      ��!       J	�`6��?�`6��?!�`6��?R      ��!       Z	�`6��?�`6��?!�`6��?JCPU_ONLYYB�'�Q@b Y      Y@q��7�w�@"�
both�Your program is POTENTIALLY input-bound because 5.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 
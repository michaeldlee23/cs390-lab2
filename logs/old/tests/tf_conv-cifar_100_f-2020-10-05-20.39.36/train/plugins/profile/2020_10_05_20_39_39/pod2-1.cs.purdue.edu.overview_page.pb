�	29�3L�a@29�3L�a@!29�3L�a@	���~5�?���~5�?!���~5�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$29�3L�a@�ǵ�b��?AW@\�a@Yٗl<�b�?*	�v��oP@2F
Iterator::Model����P1�?!$a�DmF@)bod��?1֯���=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat^�zk`��?!�Q��ƴ>@)C�K��?1T��r�8@:Preprocessing2U
Iterator::Model::ParallelMapV2Ƣ��dp�?!�0[9'].@)Ƣ��dp�?1�0[9'].@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��e��?!S�?]K2@)gF?N�{?1LM���$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice/���ިu?!Z��G� @)/���ިu?1Z��G� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��}��?!�۞]��K@)�p>?�p?1�?]K�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor� �	�o?!��y�Rm@)� �	�o?1��y�Rm@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9���~5�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ǵ�b��?�ǵ�b��?!�ǵ�b��?      ��!       "      ��!       *      ��!       2	W@\�a@W@\�a@!W@\�a@:      ��!       B      ��!       J	ٗl<�b�?ٗl<�b�?!ٗl<�b�?R      ��!       Z	ٗl<�b�?ٗl<�b�?!ٗl<�b�?JCPU_ONLYY���~5�?b Y      Y@q@��)=p�?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 
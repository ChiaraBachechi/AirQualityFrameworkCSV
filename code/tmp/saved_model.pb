¢0
Û¬
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ë.
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm_3/lstm_cell_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,**
shared_namelstm_3/lstm_cell_6/kernel

-lstm_3/lstm_cell_6/kernel/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_6/kernel*
_output_shapes

:,*
dtype0
¢
#lstm_3/lstm_cell_6/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,*4
shared_name%#lstm_3/lstm_cell_6/recurrent_kernel

7lstm_3/lstm_cell_6/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_3/lstm_cell_6/recurrent_kernel*
_output_shapes

:,*
dtype0

lstm_3/lstm_cell_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*(
shared_namelstm_3/lstm_cell_6/bias

+lstm_3/lstm_cell_6/bias/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_6/bias*
_output_shapes
:,*
dtype0

lstm_4/lstm_cell_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namelstm_4/lstm_cell_7/kernel

-lstm_4/lstm_cell_7/kernel/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_7/kernel*
_output_shapes

:*
dtype0
¢
#lstm_4/lstm_cell_7/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#lstm_4/lstm_cell_7/recurrent_kernel

7lstm_4/lstm_cell_7/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_4/lstm_cell_7/recurrent_kernel*
_output_shapes

:*
dtype0

lstm_4/lstm_cell_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namelstm_4/lstm_cell_7/bias

+lstm_4/lstm_cell_7/bias/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_7/bias*
_output_shapes
:*
dtype0

lstm_5/lstm_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namelstm_5/lstm_cell_8/kernel

-lstm_5/lstm_cell_8/kernel/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_8/kernel*
_output_shapes

:*
dtype0
¢
#lstm_5/lstm_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#lstm_5/lstm_cell_8/recurrent_kernel

7lstm_5/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_5/lstm_cell_8/recurrent_kernel*
_output_shapes

:*
dtype0

lstm_5/lstm_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namelstm_5/lstm_cell_8/bias

+lstm_5/lstm_cell_8/bias/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_8/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0

 Adam/lstm_3/lstm_cell_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,*1
shared_name" Adam/lstm_3/lstm_cell_6/kernel/m

4Adam/lstm_3/lstm_cell_6/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_6/kernel/m*
_output_shapes

:,*
dtype0
°
*Adam/lstm_3/lstm_cell_6/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,*;
shared_name,*Adam/lstm_3/lstm_cell_6/recurrent_kernel/m
©
>Adam/lstm_3/lstm_cell_6/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_6/recurrent_kernel/m*
_output_shapes

:,*
dtype0

Adam/lstm_3/lstm_cell_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*/
shared_name Adam/lstm_3/lstm_cell_6/bias/m

2Adam/lstm_3/lstm_cell_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_6/bias/m*
_output_shapes
:,*
dtype0

 Adam/lstm_4/lstm_cell_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Adam/lstm_4/lstm_cell_7/kernel/m

4Adam/lstm_4/lstm_cell_7/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_7/kernel/m*
_output_shapes

:*
dtype0
°
*Adam/lstm_4/lstm_cell_7/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/lstm_4/lstm_cell_7/recurrent_kernel/m
©
>Adam/lstm_4/lstm_cell_7/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_7/recurrent_kernel/m*
_output_shapes

:*
dtype0

Adam/lstm_4/lstm_cell_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_4/lstm_cell_7/bias/m

2Adam/lstm_4/lstm_cell_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_7/bias/m*
_output_shapes
:*
dtype0

 Adam/lstm_5/lstm_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Adam/lstm_5/lstm_cell_8/kernel/m

4Adam/lstm_5/lstm_cell_8/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_5/lstm_cell_8/kernel/m*
_output_shapes

:*
dtype0
°
*Adam/lstm_5/lstm_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/lstm_5/lstm_cell_8/recurrent_kernel/m
©
>Adam/lstm_5/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_5/lstm_cell_8/recurrent_kernel/m*
_output_shapes

:*
dtype0

Adam/lstm_5/lstm_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_5/lstm_cell_8/bias/m

2Adam/lstm_5/lstm_cell_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_5/lstm_cell_8/bias/m*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

 Adam/lstm_3/lstm_cell_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,*1
shared_name" Adam/lstm_3/lstm_cell_6/kernel/v

4Adam/lstm_3/lstm_cell_6/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_6/kernel/v*
_output_shapes

:,*
dtype0
°
*Adam/lstm_3/lstm_cell_6/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,*;
shared_name,*Adam/lstm_3/lstm_cell_6/recurrent_kernel/v
©
>Adam/lstm_3/lstm_cell_6/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_6/recurrent_kernel/v*
_output_shapes

:,*
dtype0

Adam/lstm_3/lstm_cell_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:,*/
shared_name Adam/lstm_3/lstm_cell_6/bias/v

2Adam/lstm_3/lstm_cell_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_6/bias/v*
_output_shapes
:,*
dtype0

 Adam/lstm_4/lstm_cell_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Adam/lstm_4/lstm_cell_7/kernel/v

4Adam/lstm_4/lstm_cell_7/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_7/kernel/v*
_output_shapes

:*
dtype0
°
*Adam/lstm_4/lstm_cell_7/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/lstm_4/lstm_cell_7/recurrent_kernel/v
©
>Adam/lstm_4/lstm_cell_7/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_7/recurrent_kernel/v*
_output_shapes

:*
dtype0

Adam/lstm_4/lstm_cell_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_4/lstm_cell_7/bias/v

2Adam/lstm_4/lstm_cell_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_7/bias/v*
_output_shapes
:*
dtype0

 Adam/lstm_5/lstm_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Adam/lstm_5/lstm_cell_8/kernel/v

4Adam/lstm_5/lstm_cell_8/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_5/lstm_cell_8/kernel/v*
_output_shapes

:*
dtype0
°
*Adam/lstm_5/lstm_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/lstm_5/lstm_cell_8/recurrent_kernel/v
©
>Adam/lstm_5/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_5/lstm_cell_8/recurrent_kernel/v*
_output_shapes

:*
dtype0

Adam/lstm_5/lstm_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_5/lstm_cell_8/bias/v

2Adam/lstm_5/lstm_cell_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_5/lstm_cell_8/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
²Z
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*íY
valueãYBàY BÙY

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
¥
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
* &call_and_return_all_conditional_losses* 
Á
!cell
"
state_spec
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'_random_generator
(__call__
*)&call_and_return_all_conditional_losses*
¥
*	variables
+trainable_variables
,regularization_losses
-	keras_api
._random_generator
/__call__
*0&call_and_return_all_conditional_losses* 
Á
1cell
2
state_spec
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7_random_generator
8__call__
*9&call_and_return_all_conditional_losses*
¥
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>_random_generator
?__call__
*@&call_and_return_all_conditional_losses* 
¦

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses*
 
Iiter

Jbeta_1

Kbeta_2
	Ldecay
Mlearning_rateAm¯Bm°Nm±Om²Pm³Qm´RmµSm¶Tm·Um¸Vm¹AvºBv»Nv¼Ov½Pv¾Qv¿RvÀSvÁTvÂUvÃVvÄ*
R
N0
O1
P2
Q3
R4
S5
T6
U7
V8
A9
B10*
R
N0
O1
P2
Q3
R4
S5
T6
U7
V8
A9
B10*
* 
°
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

\serving_default* 
ã
]
state_size

Nkernel
Orecurrent_kernel
Pbias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b_random_generator
c__call__
*d&call_and_return_all_conditional_losses*
* 

N0
O1
P2*

N0
O1
P2*
* 


estates
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses* 
* 
* 
* 
ã
p
state_size

Qkernel
Rrecurrent_kernel
Sbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u_random_generator
v__call__
*w&call_and_return_all_conditional_losses*
* 

Q0
R1
S2*

Q0
R1
S2*
* 


xstates
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
#	variables
$trainable_variables
%regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 
* 
* 
* 
ë

state_size

Tkernel
Urecurrent_kernel
Vbias
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses*
* 

T0
U1
V2*

T0
U1
V2*
* 
¥
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
:	variables
;trainable_variables
<regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

A0
B1*

A0
B1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_3/lstm_cell_6/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_3/lstm_cell_6/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_3/lstm_cell_6/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_4/lstm_cell_7/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_4/lstm_cell_7/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_4/lstm_cell_7/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_5/lstm_cell_8/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_5/lstm_cell_8/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_5/lstm_cell_8/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*

0*
* 
* 
* 
* 

N0
O1
P2*

N0
O1
P2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
^	variables
_trainable_variables
`regularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 

Q0
R1
S2*

Q0
R1
S2*
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
q	variables
rtrainable_variables
sregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

!0*
* 
* 
* 
* 
* 
* 
* 
* 
* 

T0
U1
V2*

T0
U1
V2*
* 

¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

10*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

«total

¬count
­	variables
®	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

«0
¬1*

­	variables*
{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_3/lstm_cell_6/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_3/lstm_cell_6/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_3/lstm_cell_6/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_4/lstm_cell_7/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_4/lstm_cell_7/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_4/lstm_cell_7/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_5/lstm_cell_8/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_5/lstm_cell_8/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_5/lstm_cell_8/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_3/lstm_cell_6/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_3/lstm_cell_6/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_3/lstm_cell_6/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_4/lstm_cell_7/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_4/lstm_cell_7/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_4/lstm_cell_7/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_5/lstm_cell_8/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_5/lstm_cell_8/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_5/lstm_cell_8/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_lstm_3_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
õ
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_3_inputlstm_3/lstm_cell_6/kernel#lstm_3/lstm_cell_6/recurrent_kernellstm_3/lstm_cell_6/biaslstm_4/lstm_cell_7/kernel#lstm_4/lstm_cell_7/recurrent_kernellstm_4/lstm_cell_7/biaslstm_5/lstm_cell_8/kernel#lstm_5/lstm_cell_8/recurrent_kernellstm_5/lstm_cell_8/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_37397
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ê
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_3/lstm_cell_6/kernel/Read/ReadVariableOp7lstm_3/lstm_cell_6/recurrent_kernel/Read/ReadVariableOp+lstm_3/lstm_cell_6/bias/Read/ReadVariableOp-lstm_4/lstm_cell_7/kernel/Read/ReadVariableOp7lstm_4/lstm_cell_7/recurrent_kernel/Read/ReadVariableOp+lstm_4/lstm_cell_7/bias/Read/ReadVariableOp-lstm_5/lstm_cell_8/kernel/Read/ReadVariableOp7lstm_5/lstm_cell_8/recurrent_kernel/Read/ReadVariableOp+lstm_5/lstm_cell_8/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_6/kernel/m/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_6/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_6/bias/m/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_7/kernel/m/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_7/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_7/bias/m/Read/ReadVariableOp4Adam/lstm_5/lstm_cell_8/kernel/m/Read/ReadVariableOp>Adam/lstm_5/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_5/lstm_cell_8/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_6/kernel/v/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_6/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_6/bias/v/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_7/kernel/v/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_7/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_7/bias/v/Read/ReadVariableOp4Adam/lstm_5/lstm_cell_8/kernel/v/Read/ReadVariableOp>Adam/lstm_5/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_5/lstm_cell_8/bias/v/Read/ReadVariableOpConst*5
Tin.
,2*	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_39782
¥
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_3/lstm_cell_6/kernel#lstm_3/lstm_cell_6/recurrent_kernellstm_3/lstm_cell_6/biaslstm_4/lstm_cell_7/kernel#lstm_4/lstm_cell_7/recurrent_kernellstm_4/lstm_cell_7/biaslstm_5/lstm_cell_8/kernel#lstm_5/lstm_cell_8/recurrent_kernellstm_5/lstm_cell_8/biastotalcountAdam/dense_1/kernel/mAdam/dense_1/bias/m Adam/lstm_3/lstm_cell_6/kernel/m*Adam/lstm_3/lstm_cell_6/recurrent_kernel/mAdam/lstm_3/lstm_cell_6/bias/m Adam/lstm_4/lstm_cell_7/kernel/m*Adam/lstm_4/lstm_cell_7/recurrent_kernel/mAdam/lstm_4/lstm_cell_7/bias/m Adam/lstm_5/lstm_cell_8/kernel/m*Adam/lstm_5/lstm_cell_8/recurrent_kernel/mAdam/lstm_5/lstm_cell_8/bias/mAdam/dense_1/kernel/vAdam/dense_1/bias/v Adam/lstm_3/lstm_cell_6/kernel/v*Adam/lstm_3/lstm_cell_6/recurrent_kernel/vAdam/lstm_3/lstm_cell_6/bias/v Adam/lstm_4/lstm_cell_7/kernel/v*Adam/lstm_4/lstm_cell_7/recurrent_kernel/vAdam/lstm_4/lstm_cell_7/bias/v Adam/lstm_5/lstm_cell_8/kernel/v*Adam/lstm_5/lstm_cell_8/recurrent_kernel/vAdam/lstm_5/lstm_cell_8/bias/v*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_39912¨´,
Í7
À
while_body_38786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_8_matmul_readvariableop_resource_0:F
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:A
3while_lstm_cell_8_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_8_matmul_readvariableop_resource:D
2while_lstm_cell_8_matmul_1_readvariableop_resource:?
1while_lstm_cell_8_biasadd_readvariableop_resource:¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0·
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0£
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ßI

A__inference_lstm_5_layer_call_and_return_conditional_losses_39299

inputs<
*lstm_cell_8_matmul_readvariableop_resource:>
,lstm_cell_8_matmul_1_readvariableop_resource:9
+lstm_cell_8_biasadd_readvariableop_resource:
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39215*
condR
while_cond_39214*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î7
ù
A__inference_lstm_5_layer_call_and_return_conditional_losses_34953

inputs#
lstm_cell_8_34871:#
lstm_cell_8_34873:
lstm_cell_8_34875:
identity¢#lstm_cell_8/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskì
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_34871lstm_cell_8_34873lstm_cell_8_34875*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_34870n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¯
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_34871lstm_cell_8_34873lstm_cell_8_34875*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_34884*
condR
while_cond_34883*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
NoOpNoOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
$
©
G__inference_sequential_1_layer_call_and_return_conditional_losses_36309

inputs
lstm_3_36279:,
lstm_3_36281:,
lstm_3_36283:,
lstm_4_36287:
lstm_4_36289:
lstm_4_36291:
lstm_5_36295:
lstm_5_36297:
lstm_5_36299:
dense_1_36303:
dense_1_36305:
identity¢dense_1/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢lstm_3/StatefulPartitionedCall¢lstm_4/StatefulPartitionedCall¢lstm_5/StatefulPartitionedCallù
lstm_3/StatefulPartitionedCallStatefulPartitionedCallinputslstm_3_36279lstm_3_36281lstm_3_36283*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_36238î
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_36079
lstm_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0lstm_4_36287lstm_4_36289lstm_4_36291*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_36050
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_35891
lstm_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0lstm_5_36295lstm_5_36297lstm_5_36299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_35862
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_35703
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_1_36303dense_1_36305*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_35641w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp ^dense_1/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
¾
while_cond_35531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_35531___redundant_placeholder03
/while_while_cond_35531___redundant_placeholder13
/while_while_cond_35531___redundant_placeholder23
/while_while_cond_35531___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

²
&__inference_lstm_5_layer_call_fn_38694
inputs_0
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_34953o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÚI

A__inference_lstm_3_layer_call_and_return_conditional_losses_36238

inputs<
*lstm_cell_6_matmul_readvariableop_resource:,>
,lstm_cell_6_matmul_1_readvariableop_resource:,9
+lstm_cell_6_biasadd_readvariableop_resource:,
identity¢"lstm_cell_6/BiasAdd/ReadVariableOp¢!lstm_cell_6/MatMul/ReadVariableOp¢#lstm_cell_6/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_6/MatMul/ReadVariableOpReadVariableOp*lstm_cell_6_matmul_readvariableop_resource*
_output_shapes

:,*
dtype0
lstm_cell_6/MatMulMatMulstrided_slice_2:output:0)lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
#lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0
lstm_cell_6/MatMul_1MatMulzeros:output:0+lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
lstm_cell_6/addAddV2lstm_cell_6/MatMul:product:0lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
"lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0
lstm_cell_6/BiasAddBiasAddlstm_cell_6/add:z:0*lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,]
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_6/SigmoidSigmoidlstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_6/mulMullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_6/ReluRelulstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_6/mul_1Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_6/add_1AddV2lstm_cell_6/mul:z:0lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_6/Relu_1Relulstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_6/mul_2Mullstm_cell_6/Sigmoid_2:y:0 lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_6_matmul_readvariableop_resource,lstm_cell_6_matmul_1_readvariableop_resource+lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_36154*
condR
while_cond_36153*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_6/BiasAdd/ReadVariableOp"^lstm_cell_6/MatMul/ReadVariableOp$^lstm_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_6/BiasAdd/ReadVariableOp"lstm_cell_6/BiasAdd/ReadVariableOp2F
!lstm_cell_6/MatMul/ReadVariableOp!lstm_cell_6/MatMul/ReadVariableOp2J
#lstm_cell_6/MatMul_1/ReadVariableOp#lstm_cell_6/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
½
G__inference_sequential_1_layer_call_and_return_conditional_losses_35648

inputs
lstm_3_35303:,
lstm_3_35305:,
lstm_3_35307:,
lstm_4_35460:
lstm_4_35462:
lstm_4_35464:
lstm_5_35617:
lstm_5_35619:
lstm_5_35621:
dense_1_35642:
dense_1_35644:
identity¢dense_1/StatefulPartitionedCall¢lstm_3/StatefulPartitionedCall¢lstm_4/StatefulPartitionedCall¢lstm_5/StatefulPartitionedCallù
lstm_3/StatefulPartitionedCallStatefulPartitionedCallinputslstm_3_35303lstm_3_35305lstm_3_35307*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_35302Þ
dropout_3/PartitionedCallPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_35315
lstm_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0lstm_4_35460lstm_4_35462lstm_4_35464*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_35459Þ
dropout_4/PartitionedCallPartitionedCall'lstm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_35472
lstm_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0lstm_5_35617lstm_5_35619lstm_5_35621*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_35616Ú
dropout_5/PartitionedCallPartitionedCall'lstm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_35629
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_1_35642dense_1_35644*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_35641w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
NoOpNoOp ^dense_1/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï?
 

lstm_5_while_body_37270*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0K
9lstm_5_while_lstm_cell_8_matmul_readvariableop_resource_0:M
;lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource_0:H
:lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource_0:
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensorI
7lstm_5_while_lstm_cell_8_matmul_readvariableop_resource:K
9lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource:F
8lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource:¢/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp¢.lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp¢0lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   É
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.lstm_5/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9lstm_5_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ì
lstm_5/while/lstm_cell_8/MatMulMatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
0lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0³
!lstm_5/while/lstm_cell_8/MatMul_1MatMullstm_5_while_placeholder_28lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
lstm_5/while/lstm_cell_8/addAddV2)lstm_5/while/lstm_cell_8/MatMul:product:0+lstm_5/while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0¸
 lstm_5/while/lstm_cell_8/BiasAddBiasAdd lstm_5/while/lstm_cell_8/add:z:07lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(lstm_5/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_5/while/lstm_cell_8/splitSplit1lstm_5/while/lstm_cell_8/split/split_dim:output:0)lstm_5/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 lstm_5/while/lstm_cell_8/SigmoidSigmoid'lstm_5/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_5/while/lstm_cell_8/Sigmoid_1Sigmoid'lstm_5/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/while/lstm_cell_8/mulMul&lstm_5/while/lstm_cell_8/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/while/lstm_cell_8/ReluRelu'lstm_5/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
lstm_5/while/lstm_cell_8/mul_1Mul$lstm_5/while/lstm_cell_8/Sigmoid:y:0+lstm_5/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/while/lstm_cell_8/add_1AddV2 lstm_5/while/lstm_cell_8/mul:z:0"lstm_5/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_5/while/lstm_cell_8/Sigmoid_2Sigmoid'lstm_5/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_5/while/lstm_cell_8/Relu_1Relu"lstm_5/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
lstm_5/while/lstm_cell_8/mul_2Mul&lstm_5/while/lstm_cell_8/Sigmoid_2:y:0-lstm_5/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder"lstm_5/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0^lstm_5/while/NoOp*
T0*
_output_shapes
: 
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations^lstm_5/while/NoOp*
T0*
_output_shapes
: n
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0^lstm_5/while/NoOp*
T0*
_output_shapes
: ®
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_5/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_5/while/Identity_4Identity"lstm_5/while/lstm_cell_8/mul_2:z:0^lstm_5/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/while/Identity_5Identity"lstm_5/while/lstm_cell_8/add_1:z:0^lstm_5/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
lstm_5/while/NoOpNoOp0^lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"v
8lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource:lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource_0"x
9lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource;lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource_0"t
7lstm_5_while_lstm_cell_8_matmul_readvariableop_resource9lstm_5_while_lstm_cell_8_matmul_readvariableop_resource_0"Ä
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2b
/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp2`
.lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp.lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp2d
0lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp0lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
O
À
$sequential_1_lstm_5_while_body_34012D
@sequential_1_lstm_5_while_sequential_1_lstm_5_while_loop_counterJ
Fsequential_1_lstm_5_while_sequential_1_lstm_5_while_maximum_iterations)
%sequential_1_lstm_5_while_placeholder+
'sequential_1_lstm_5_while_placeholder_1+
'sequential_1_lstm_5_while_placeholder_2+
'sequential_1_lstm_5_while_placeholder_3C
?sequential_1_lstm_5_while_sequential_1_lstm_5_strided_slice_1_0
{sequential_1_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_5_tensorarrayunstack_tensorlistfromtensor_0X
Fsequential_1_lstm_5_while_lstm_cell_8_matmul_readvariableop_resource_0:Z
Hsequential_1_lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource_0:U
Gsequential_1_lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource_0:&
"sequential_1_lstm_5_while_identity(
$sequential_1_lstm_5_while_identity_1(
$sequential_1_lstm_5_while_identity_2(
$sequential_1_lstm_5_while_identity_3(
$sequential_1_lstm_5_while_identity_4(
$sequential_1_lstm_5_while_identity_5A
=sequential_1_lstm_5_while_sequential_1_lstm_5_strided_slice_1}
ysequential_1_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_5_tensorarrayunstack_tensorlistfromtensorV
Dsequential_1_lstm_5_while_lstm_cell_8_matmul_readvariableop_resource:X
Fsequential_1_lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource:S
Esequential_1_lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource:¢<sequential_1/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp¢;sequential_1/lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp¢=sequential_1/lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp
Ksequential_1/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
=sequential_1/lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_5_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_5_while_placeholderTsequential_1/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Â
;sequential_1/lstm_5/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpFsequential_1_lstm_5_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0ó
,sequential_1/lstm_5/while/lstm_cell_8/MatMulMatMulDsequential_1/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_1/lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
=sequential_1/lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpHsequential_1_lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Ú
.sequential_1/lstm_5/while/lstm_cell_8/MatMul_1MatMul'sequential_1_lstm_5_while_placeholder_2Esequential_1/lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
)sequential_1/lstm_5/while/lstm_cell_8/addAddV26sequential_1/lstm_5/while/lstm_cell_8/MatMul:product:08sequential_1/lstm_5/while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
<sequential_1/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpGsequential_1_lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ß
-sequential_1/lstm_5/while/lstm_cell_8/BiasAddBiasAdd-sequential_1/lstm_5/while/lstm_cell_8/add:z:0Dsequential_1/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
5sequential_1/lstm_5/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¨
+sequential_1/lstm_5/while/lstm_cell_8/splitSplit>sequential_1/lstm_5/while/lstm_cell_8/split/split_dim:output:06sequential_1/lstm_5/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split 
-sequential_1/lstm_5/while/lstm_cell_8/SigmoidSigmoid4sequential_1/lstm_5/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
/sequential_1/lstm_5/while/lstm_cell_8/Sigmoid_1Sigmoid4sequential_1/lstm_5/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
)sequential_1/lstm_5/while/lstm_cell_8/mulMul3sequential_1/lstm_5/while/lstm_cell_8/Sigmoid_1:y:0'sequential_1_lstm_5_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_1/lstm_5/while/lstm_cell_8/ReluRelu4sequential_1/lstm_5/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
+sequential_1/lstm_5/while/lstm_cell_8/mul_1Mul1sequential_1/lstm_5/while/lstm_cell_8/Sigmoid:y:08sequential_1/lstm_5/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
+sequential_1/lstm_5/while/lstm_cell_8/add_1AddV2-sequential_1/lstm_5/while/lstm_cell_8/mul:z:0/sequential_1/lstm_5/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
/sequential_1/lstm_5/while/lstm_cell_8/Sigmoid_2Sigmoid4sequential_1/lstm_5/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_1/lstm_5/while/lstm_cell_8/Relu_1Relu/sequential_1/lstm_5/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
+sequential_1/lstm_5/while/lstm_cell_8/mul_2Mul3sequential_1/lstm_5/while/lstm_cell_8/Sigmoid_2:y:0:sequential_1/lstm_5/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>sequential_1/lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_5_while_placeholder_1%sequential_1_lstm_5_while_placeholder/sequential_1/lstm_5/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒa
sequential_1/lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_1/lstm_5/while/addAddV2%sequential_1_lstm_5_while_placeholder(sequential_1/lstm_5/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_1/lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :·
sequential_1/lstm_5/while/add_1AddV2@sequential_1_lstm_5_while_sequential_1_lstm_5_while_loop_counter*sequential_1/lstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_1/lstm_5/while/IdentityIdentity#sequential_1/lstm_5/while/add_1:z:0^sequential_1/lstm_5/while/NoOp*
T0*
_output_shapes
: º
$sequential_1/lstm_5/while/Identity_1IdentityFsequential_1_lstm_5_while_sequential_1_lstm_5_while_maximum_iterations^sequential_1/lstm_5/while/NoOp*
T0*
_output_shapes
: 
$sequential_1/lstm_5/while/Identity_2Identity!sequential_1/lstm_5/while/add:z:0^sequential_1/lstm_5/while/NoOp*
T0*
_output_shapes
: Õ
$sequential_1/lstm_5/while/Identity_3IdentityNsequential_1/lstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/lstm_5/while/NoOp*
T0*
_output_shapes
: :éèÒ´
$sequential_1/lstm_5/while/Identity_4Identity/sequential_1/lstm_5/while/lstm_cell_8/mul_2:z:0^sequential_1/lstm_5/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
$sequential_1/lstm_5/while/Identity_5Identity/sequential_1/lstm_5/while/lstm_cell_8/add_1:z:0^sequential_1/lstm_5/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_1/lstm_5/while/NoOpNoOp=^sequential_1/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp<^sequential_1/lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp>^sequential_1/lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_1_lstm_5_while_identity+sequential_1/lstm_5/while/Identity:output:0"U
$sequential_1_lstm_5_while_identity_1-sequential_1/lstm_5/while/Identity_1:output:0"U
$sequential_1_lstm_5_while_identity_2-sequential_1/lstm_5/while/Identity_2:output:0"U
$sequential_1_lstm_5_while_identity_3-sequential_1/lstm_5/while/Identity_3:output:0"U
$sequential_1_lstm_5_while_identity_4-sequential_1/lstm_5/while/Identity_4:output:0"U
$sequential_1_lstm_5_while_identity_5-sequential_1/lstm_5/while/Identity_5:output:0"
Esequential_1_lstm_5_while_lstm_cell_8_biasadd_readvariableop_resourceGsequential_1_lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource_0"
Fsequential_1_lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resourceHsequential_1_lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource_0"
Dsequential_1_lstm_5_while_lstm_cell_8_matmul_readvariableop_resourceFsequential_1_lstm_5_while_lstm_cell_8_matmul_readvariableop_resource_0"
=sequential_1_lstm_5_while_sequential_1_lstm_5_strided_slice_1?sequential_1_lstm_5_while_sequential_1_lstm_5_strided_slice_1_0"ø
ysequential_1_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_5_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2|
<sequential_1/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp<sequential_1/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp2z
;sequential_1/lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp;sequential_1/lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp2~
=sequential_1/lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp=sequential_1/lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ð
°
&__inference_lstm_5_layer_call_fn_38727

inputs
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_35862o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_39314

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò7
ù
A__inference_lstm_3_layer_call_and_return_conditional_losses_34444

inputs#
lstm_cell_6_34362:,#
lstm_cell_6_34364:,
lstm_cell_6_34366:,
identity¢#lstm_cell_6/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskì
#lstm_cell_6/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_6_34362lstm_cell_6_34364lstm_cell_6_34366*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_34316n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¯
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_6_34362lstm_cell_6_34364lstm_cell_6_34366*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_34375*
condR
while_cond_34374*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿt
NoOpNoOp$^lstm_cell_6/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_6/StatefulPartitionedCall#lstm_cell_6/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê"
Ï
while_body_34375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_6_34399_0:,+
while_lstm_cell_6_34401_0:,'
while_lstm_cell_6_34403_0:,
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_6_34399:,)
while_lstm_cell_6_34401:,%
while_lstm_cell_6_34403:,¢)while/lstm_cell_6/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0ª
)while/lstm_cell_6/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_6_34399_0while_lstm_cell_6_34401_0while_lstm_cell_6_34403_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_34316Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_6/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/lstm_cell_6/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity2while/lstm_cell_6/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

while/NoOpNoOp*^while/lstm_cell_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_6_34399while_lstm_cell_6_34399_0"4
while_lstm_cell_6_34401while_lstm_cell_6_34401_0"4
while_lstm_cell_6_34403while_lstm_cell_6_34403_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_6/StatefulPartitionedCall)while/lstm_cell_6/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
þ	
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_39326

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï?
 

lstm_3_while_body_36976*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0K
9lstm_3_while_lstm_cell_6_matmul_readvariableop_resource_0:,M
;lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource_0:,H
:lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource_0:,
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorI
7lstm_3_while_lstm_cell_6_matmul_readvariableop_resource:,K
9lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource:,F
8lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource:,¢/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp¢.lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp¢0lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   É
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.lstm_3/while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp9lstm_3_while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:,*
dtype0Ì
lstm_3/while/lstm_cell_6/MatMulMatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,¬
0lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp;lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:,*
dtype0³
!lstm_3/while/lstm_cell_6/MatMul_1MatMullstm_3_while_placeholder_28lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,¯
lstm_3/while/lstm_cell_6/addAddV2)lstm_3/while/lstm_cell_6/MatMul:product:0+lstm_3/while/lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,¦
/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:,*
dtype0¸
 lstm_3/while/lstm_cell_6/BiasAddBiasAdd lstm_3/while/lstm_cell_6/add:z:07lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,j
(lstm_3/while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_3/while/lstm_cell_6/splitSplit1lstm_3/while/lstm_cell_6/split/split_dim:output:0)lstm_3/while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 lstm_3/while/lstm_cell_6/SigmoidSigmoid'lstm_3/while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_3/while/lstm_cell_6/Sigmoid_1Sigmoid'lstm_3/while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/while/lstm_cell_6/mulMul&lstm_3/while/lstm_cell_6/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/while/lstm_cell_6/ReluRelu'lstm_3/while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
lstm_3/while/lstm_cell_6/mul_1Mul$lstm_3/while/lstm_cell_6/Sigmoid:y:0+lstm_3/while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/while/lstm_cell_6/add_1AddV2 lstm_3/while/lstm_cell_6/mul:z:0"lstm_3/while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_3/while/lstm_cell_6/Sigmoid_2Sigmoid'lstm_3/while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_3/while/lstm_cell_6/Relu_1Relu"lstm_3/while/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
lstm_3/while/lstm_cell_6/mul_2Mul&lstm_3/while/lstm_cell_6/Sigmoid_2:y:0-lstm_3/while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder"lstm_3/while/lstm_cell_6/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations^lstm_3/while/NoOp*
T0*
_output_shapes
: n
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: ®
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_3/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_3/while/Identity_4Identity"lstm_3/while/lstm_cell_6/mul_2:z:0^lstm_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_6/add_1:z:0^lstm_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
lstm_3/while/NoOpNoOp0^lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"v
8lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource:lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource_0"x
9lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource;lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource_0"t
7lstm_3_while_lstm_cell_6_matmul_readvariableop_resource9lstm_3_while_lstm_cell_6_matmul_readvariableop_resource_0"Ä
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2b
/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp2`
.lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp.lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp2d
0lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp0lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ô	
Ê
lstm_5_while_cond_36825*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1A
=lstm_5_while_lstm_5_while_cond_36825___redundant_placeholder0A
=lstm_5_while_lstm_5_while_cond_36825___redundant_placeholder1A
=lstm_5_while_lstm_5_while_cond_36825___redundant_placeholder2A
=lstm_5_while_lstm_5_while_cond_36825___redundant_placeholder3
lstm_5_while_identity
~
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: Y
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_5_while_identitylstm_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
°
¾
while_cond_37928
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_37928___redundant_placeholder03
/while_while_cond_37928___redundant_placeholder13
/while_while_cond_37928___redundant_placeholder23
/while_while_cond_37928___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ä
ñ
+__inference_lstm_cell_8_layer_call_fn_39575

inputs
states_0
states_1
unknown:
	unknown_0:
	unknown_1:
identity

identity_1

identity_2¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_35016o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ÚI

A__inference_lstm_4_layer_call_and_return_conditional_losses_38513

inputs<
*lstm_cell_7_matmul_readvariableop_resource:>
,lstm_cell_7_matmul_1_readvariableop_resource:9
+lstm_cell_7_biasadd_readvariableop_resource:
identity¢"lstm_cell_7/BiasAdd/ReadVariableOp¢!lstm_cell_7/MatMul/ReadVariableOp¢#lstm_cell_7/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_38429*
condR
while_cond_38428*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_7/BiasAdd/ReadVariableOp"^lstm_cell_7/MatMul/ReadVariableOp$^lstm_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_7/BiasAdd/ReadVariableOp"lstm_cell_7/BiasAdd/ReadVariableOp2F
!lstm_cell_7/MatMul/ReadVariableOp!lstm_cell_7/MatMul/ReadVariableOp2J
#lstm_cell_7/MatMul_1/ReadVariableOp#lstm_cell_7/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í7
À
while_body_39072
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_8_matmul_readvariableop_resource_0:F
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:A
3while_lstm_cell_8_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_8_matmul_readvariableop_resource:D
2while_lstm_cell_8_matmul_1_readvariableop_resource:?
1while_lstm_cell_8_biasadd_readvariableop_resource:¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0·
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0£
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°
¾
while_cond_37642
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_37642___redundant_placeholder03
/while_while_cond_37642___redundant_placeholder13
/while_while_cond_37642___redundant_placeholder23
/while_while_cond_37642___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ê"
Ï
while_body_34534
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_7_34558_0:+
while_lstm_cell_7_34560_0:'
while_lstm_cell_7_34562_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_7_34558:)
while_lstm_cell_7_34560:%
while_lstm_cell_7_34562:¢)while/lstm_cell_7/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0ª
)while/lstm_cell_7/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_7_34558_0while_lstm_cell_7_34560_0while_lstm_cell_7_34562_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_34520Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_7/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/lstm_cell_7/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity2while/lstm_cell_7/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

while/NoOpNoOp*^while/lstm_cell_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_7_34558while_lstm_cell_7_34558_0"4
while_lstm_cell_7_34560while_lstm_cell_7_34560_0"4
while_lstm_cell_7_34562while_lstm_cell_7_34562_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_7/StatefulPartitionedCall)while/lstm_cell_7/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ï?
 

lstm_5_while_body_36826*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0K
9lstm_5_while_lstm_cell_8_matmul_readvariableop_resource_0:M
;lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource_0:H
:lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource_0:
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensorI
7lstm_5_while_lstm_cell_8_matmul_readvariableop_resource:K
9lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource:F
8lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource:¢/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp¢.lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp¢0lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   É
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.lstm_5/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9lstm_5_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ì
lstm_5/while/lstm_cell_8/MatMulMatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
0lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0³
!lstm_5/while/lstm_cell_8/MatMul_1MatMullstm_5_while_placeholder_28lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
lstm_5/while/lstm_cell_8/addAddV2)lstm_5/while/lstm_cell_8/MatMul:product:0+lstm_5/while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0¸
 lstm_5/while/lstm_cell_8/BiasAddBiasAdd lstm_5/while/lstm_cell_8/add:z:07lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(lstm_5/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_5/while/lstm_cell_8/splitSplit1lstm_5/while/lstm_cell_8/split/split_dim:output:0)lstm_5/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 lstm_5/while/lstm_cell_8/SigmoidSigmoid'lstm_5/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_5/while/lstm_cell_8/Sigmoid_1Sigmoid'lstm_5/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/while/lstm_cell_8/mulMul&lstm_5/while/lstm_cell_8/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/while/lstm_cell_8/ReluRelu'lstm_5/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
lstm_5/while/lstm_cell_8/mul_1Mul$lstm_5/while/lstm_cell_8/Sigmoid:y:0+lstm_5/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/while/lstm_cell_8/add_1AddV2 lstm_5/while/lstm_cell_8/mul:z:0"lstm_5/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_5/while/lstm_cell_8/Sigmoid_2Sigmoid'lstm_5/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_5/while/lstm_cell_8/Relu_1Relu"lstm_5/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
lstm_5/while/lstm_cell_8/mul_2Mul&lstm_5/while/lstm_cell_8/Sigmoid_2:y:0-lstm_5/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder"lstm_5/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0^lstm_5/while/NoOp*
T0*
_output_shapes
: 
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations^lstm_5/while/NoOp*
T0*
_output_shapes
: n
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0^lstm_5/while/NoOp*
T0*
_output_shapes
: ®
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_5/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_5/while/Identity_4Identity"lstm_5/while/lstm_cell_8/mul_2:z:0^lstm_5/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/while/Identity_5Identity"lstm_5/while/lstm_cell_8/add_1:z:0^lstm_5/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
lstm_5/while/NoOpNoOp0^lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"v
8lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource:lstm_5_while_lstm_cell_8_biasadd_readvariableop_resource_0"x
9lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource;lstm_5_while_lstm_cell_8_matmul_1_readvariableop_resource_0"t
7lstm_5_while_lstm_cell_8_matmul_readvariableop_resource9lstm_5_while_lstm_cell_8_matmul_readvariableop_resource_0"Ä
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2b
/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp/lstm_5/while/lstm_cell_8/BiasAdd/ReadVariableOp2`
.lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp.lstm_5/while/lstm_cell_8/MatMul/ReadVariableOp2d
0lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp0lstm_5/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¢
²
&__inference_lstm_3_layer_call_fn_37408
inputs_0
unknown:,
	unknown_0:,
	unknown_1:,
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_34253|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÿ
b
)__inference_dropout_3_layer_call_fn_38023

inputs
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_36079s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÚI

A__inference_lstm_3_layer_call_and_return_conditional_losses_35302

inputs<
*lstm_cell_6_matmul_readvariableop_resource:,>
,lstm_cell_6_matmul_1_readvariableop_resource:,9
+lstm_cell_6_biasadd_readvariableop_resource:,
identity¢"lstm_cell_6/BiasAdd/ReadVariableOp¢!lstm_cell_6/MatMul/ReadVariableOp¢#lstm_cell_6/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_6/MatMul/ReadVariableOpReadVariableOp*lstm_cell_6_matmul_readvariableop_resource*
_output_shapes

:,*
dtype0
lstm_cell_6/MatMulMatMulstrided_slice_2:output:0)lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
#lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0
lstm_cell_6/MatMul_1MatMulzeros:output:0+lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
lstm_cell_6/addAddV2lstm_cell_6/MatMul:product:0lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
"lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0
lstm_cell_6/BiasAddBiasAddlstm_cell_6/add:z:0*lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,]
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_6/SigmoidSigmoidlstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_6/mulMullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_6/ReluRelulstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_6/mul_1Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_6/add_1AddV2lstm_cell_6/mul:z:0lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_6/Relu_1Relulstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_6/mul_2Mullstm_cell_6/Sigmoid_2:y:0 lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_6_matmul_readvariableop_resource,lstm_cell_6_matmul_1_readvariableop_resource+lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35218*
condR
while_cond_35217*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_6/BiasAdd/ReadVariableOp"^lstm_cell_6/MatMul/ReadVariableOp$^lstm_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_6/BiasAdd/ReadVariableOp"lstm_cell_6/BiasAdd/ReadVariableOp2F
!lstm_cell_6/MatMul/ReadVariableOp!lstm_cell_6/MatMul/ReadVariableOp2J
#lstm_cell_6/MatMul_1/ReadVariableOp#lstm_cell_6/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò7
ù
A__inference_lstm_4_layer_call_and_return_conditional_losses_34603

inputs#
lstm_cell_7_34521:#
lstm_cell_7_34523:
lstm_cell_7_34525:
identity¢#lstm_cell_7/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskì
#lstm_cell_7/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_7_34521lstm_cell_7_34523lstm_cell_7_34525*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_34520n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¯
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_7_34521lstm_cell_7_34523lstm_cell_7_34525*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_34534*
condR
while_cond_34533*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿt
NoOpNoOp$^lstm_cell_7/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_7/StatefulPartitionedCall#lstm_cell_7/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
²
&__inference_lstm_3_layer_call_fn_37419
inputs_0
unknown:,
	unknown_0:,
	unknown_1:,
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_34444|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Í7
À
while_body_39215
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_8_matmul_readvariableop_resource_0:F
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:A
3while_lstm_cell_8_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_8_matmul_readvariableop_resource:D
2while_lstm_cell_8_matmul_1_readvariableop_resource:?
1while_lstm_cell_8_biasadd_readvariableop_resource:¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0·
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0£
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°
¾
while_cond_37499
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_37499___redundant_placeholder03
/while_while_cond_37499___redundant_placeholder13
/while_while_cond_37499___redundant_placeholder23
/while_while_cond_37499___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ä
ñ
+__inference_lstm_cell_7_layer_call_fn_39477

inputs
states_0
states_1
unknown:
	unknown_0:
	unknown_1:
identity

identity_1

identity_2¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_34666o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
²


#__inference_signature_wrapper_37397
lstm_3_input
unknown:,
	unknown_0:,
	unknown_1:,
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCalllstm_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_34103o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_3_input
Í7
À
while_body_35966
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_7_matmul_readvariableop_resource_0:F
4while_lstm_cell_7_matmul_1_readvariableop_resource_0:A
3while_lstm_cell_7_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_7_matmul_readvariableop_resource:D
2while_lstm_cell_7_matmul_1_readvariableop_resource:?
1while_lstm_cell_7_biasadd_readvariableop_resource:¢(while/lstm_cell_7/BiasAdd/ReadVariableOp¢'while/lstm_cell_7/MatMul/ReadVariableOp¢)while/lstm_cell_7/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0·
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0£
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_7/BiasAdd/ReadVariableOp(^while/lstm_cell_7/MatMul/ReadVariableOp*^while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_7/BiasAdd/ReadVariableOp(while/lstm_cell_7/BiasAdd/ReadVariableOp2R
'while/lstm_cell_7/MatMul/ReadVariableOp'while/lstm_cell_7/MatMul/ReadVariableOp2V
)while/lstm_cell_7/MatMul_1/ReadVariableOp)while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°
¾
while_cond_34374
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_34374___redundant_placeholder03
/while_while_cond_34374___redundant_placeholder13
/while_while_cond_34374___redundant_placeholder23
/while_while_cond_34374___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
°
¾
while_cond_34883
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_34883___redundant_placeholder03
/while_while_cond_34883___redundant_placeholder13
/while_while_cond_34883___redundant_placeholder23
/while_while_cond_34883___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ç
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_35315

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
ÿ
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_34170

inputs

states
states_10
matmul_readvariableop_resource:,2
 matmul_1_readvariableop_resource:,-
biasadd_readvariableop_resource:,
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:,*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Í7
À
while_body_37500
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_6_matmul_readvariableop_resource_0:,F
4while_lstm_cell_6_matmul_1_readvariableop_resource_0:,A
3while_lstm_cell_6_biasadd_readvariableop_resource_0:,
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_6_matmul_readvariableop_resource:,D
2while_lstm_cell_6_matmul_1_readvariableop_resource:,?
1while_lstm_cell_6_biasadd_readvariableop_resource:,¢(while/lstm_cell_6/BiasAdd/ReadVariableOp¢'while/lstm_cell_6/MatMul/ReadVariableOp¢)while/lstm_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:,*
dtype0·
while/lstm_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
)while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:,*
dtype0
while/lstm_cell_6/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
while/lstm_cell_6/addAddV2"while/lstm_cell_6/MatMul:product:0$while/lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
(while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:,*
dtype0£
while/lstm_cell_6/BiasAddBiasAddwhile/lstm_cell_6/add:z:00while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,c
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0"while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_6/SigmoidSigmoid while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_6/Sigmoid_1Sigmoid while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mulMulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_6/ReluRelu while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mul_1Mulwhile/lstm_cell_6/Sigmoid:y:0$while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/add_1AddV2while/lstm_cell_6/mul:z:0while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_6/Sigmoid_2Sigmoid while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_6/Relu_1Reluwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mul_2Mulwhile/lstm_cell_6/Sigmoid_2:y:0&while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_6/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_6/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_6/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_6_biasadd_readvariableop_resource3while_lstm_cell_6_biasadd_readvariableop_resource_0"j
2while_lstm_cell_6_matmul_1_readvariableop_resource4while_lstm_cell_6_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_6_matmul_readvariableop_resource2while_lstm_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_6/BiasAdd/ReadVariableOp(while/lstm_cell_6/BiasAdd/ReadVariableOp2R
'while/lstm_cell_6/MatMul/ReadVariableOp'while/lstm_cell_6/MatMul/ReadVariableOp2V
)while/lstm_cell_6/MatMul_1/ReadVariableOp)while/lstm_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

»

G__inference_sequential_1_layer_call_and_return_conditional_losses_37368

inputsC
1lstm_3_lstm_cell_6_matmul_readvariableop_resource:,E
3lstm_3_lstm_cell_6_matmul_1_readvariableop_resource:,@
2lstm_3_lstm_cell_6_biasadd_readvariableop_resource:,C
1lstm_4_lstm_cell_7_matmul_readvariableop_resource:E
3lstm_4_lstm_cell_7_matmul_1_readvariableop_resource:@
2lstm_4_lstm_cell_7_biasadd_readvariableop_resource:C
1lstm_5_lstm_cell_8_matmul_readvariableop_resource:E
3lstm_5_lstm_cell_8_matmul_1_readvariableop_resource:@
2lstm_5_lstm_cell_8_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢)lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp¢(lstm_3/lstm_cell_6/MatMul/ReadVariableOp¢*lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp¢lstm_3/while¢)lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp¢(lstm_4/lstm_cell_7/MatMul/ReadVariableOp¢*lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp¢lstm_4/while¢)lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp¢(lstm_5/lstm_cell_8/MatMul/ReadVariableOp¢*lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp¢lstm_5/whileB
lstm_3/ShapeShapeinputs*
T0*
_output_shapes
:d
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          {
lstm_3/transpose	Transposeinputslstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:f
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   õ
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(lstm_3/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp1lstm_3_lstm_cell_6_matmul_readvariableop_resource*
_output_shapes

:,*
dtype0¨
lstm_3/lstm_cell_6/MatMulMatMullstm_3/strided_slice_2:output:00lstm_3/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
*lstm_3/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp3lstm_3_lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0¢
lstm_3/lstm_cell_6/MatMul_1MatMullstm_3/zeros:output:02lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
lstm_3/lstm_cell_6/addAddV2#lstm_3/lstm_cell_6/MatMul:product:0%lstm_3/lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
)lstm_3/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0¦
lstm_3/lstm_cell_6/BiasAddBiasAddlstm_3/lstm_cell_6/add:z:01lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,d
"lstm_3/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
lstm_3/lstm_cell_6/splitSplit+lstm_3/lstm_cell_6/split/split_dim:output:0#lstm_3/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
lstm_3/lstm_cell_6/SigmoidSigmoid!lstm_3/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_3/lstm_cell_6/Sigmoid_1Sigmoid!lstm_3/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/lstm_cell_6/mulMul lstm_3/lstm_cell_6/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm_3/lstm_cell_6/ReluRelu!lstm_3/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/lstm_cell_6/mul_1Mullstm_3/lstm_cell_6/Sigmoid:y:0%lstm_3/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/lstm_cell_6/add_1AddV2lstm_3/lstm_cell_6/mul:z:0lstm_3/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_3/lstm_cell_6/Sigmoid_2Sigmoid!lstm_3/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_3/lstm_cell_6/Relu_1Relulstm_3/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/lstm_cell_6/mul_2Mul lstm_3/lstm_cell_6/Sigmoid_2:y:0'lstm_3/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Í
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ß
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_3_lstm_cell_6_matmul_readvariableop_resource3lstm_3_lstm_cell_6_matmul_1_readvariableop_resource2lstm_3_lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_3_while_body_36976*#
condR
lstm_3_while_cond_36975*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0o
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskl
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_3/dropout/MulMullstm_3/transpose_1:y:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout_3/dropout/ShapeShapelstm_3/transpose_1:y:0*
T0*
_output_shapes
:°
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seede
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=È
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
lstm_4/ShapeShapedropout_3/dropout/Mul_1:z:0*
T0*
_output_shapes
:d
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_4/transpose	Transposedropout_3/dropout/Mul_1:z:0lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
lstm_4/Shape_1Shapelstm_4/transpose:y:0*
T0*
_output_shapes
:f
lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_4/strided_slice_1StridedSlicelstm_4/Shape_1:output:0%lstm_4/strided_slice_1/stack:output:0'lstm_4/strided_slice_1/stack_1:output:0'lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_4/TensorArrayV2TensorListReserve+lstm_4/TensorArrayV2/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   õ
.lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_4/transpose:y:0Elstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_4/strided_slice_2StridedSlicelstm_4/transpose:y:0%lstm_4/strided_slice_2/stack:output:0'lstm_4/strided_slice_2/stack_1:output:0'lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(lstm_4/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp1lstm_4_lstm_cell_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¨
lstm_4/lstm_cell_7/MatMulMatMullstm_4/strided_slice_2:output:00lstm_4/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_4/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp3lstm_4_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0¢
lstm_4/lstm_cell_7/MatMul_1MatMullstm_4/zeros:output:02lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/lstm_cell_7/addAddV2#lstm_4/lstm_cell_7/MatMul:product:0%lstm_4/lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)lstm_4/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
lstm_4/lstm_cell_7/BiasAddBiasAddlstm_4/lstm_cell_7/add:z:01lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"lstm_4/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
lstm_4/lstm_cell_7/splitSplit+lstm_4/lstm_cell_7/split/split_dim:output:0#lstm_4/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
lstm_4/lstm_cell_7/SigmoidSigmoid!lstm_4/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_4/lstm_cell_7/Sigmoid_1Sigmoid!lstm_4/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/lstm_cell_7/mulMul lstm_4/lstm_cell_7/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm_4/lstm_cell_7/ReluRelu!lstm_4/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/lstm_cell_7/mul_1Mullstm_4/lstm_cell_7/Sigmoid:y:0%lstm_4/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/lstm_cell_7/add_1AddV2lstm_4/lstm_cell_7/mul:z:0lstm_4/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_4/lstm_cell_7/Sigmoid_2Sigmoid!lstm_4/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_4/lstm_cell_7/Relu_1Relulstm_4/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/lstm_cell_7/mul_2Mul lstm_4/lstm_cell_7/Sigmoid_2:y:0'lstm_4/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Í
lstm_4/TensorArrayV2_1TensorListReserve-lstm_4/TensorArrayV2_1/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ß
lstm_4/whileWhile"lstm_4/while/loop_counter:output:0(lstm_4/while/maximum_iterations:output:0lstm_4/time:output:0lstm_4/TensorArrayV2_1:handle:0lstm_4/zeros:output:0lstm_4/zeros_1:output:0lstm_4/strided_slice_1:output:0>lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_4_lstm_cell_7_matmul_readvariableop_resource3lstm_4_lstm_cell_7_matmul_1_readvariableop_resource2lstm_4_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_4_while_body_37123*#
condR
lstm_4_while_cond_37122*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0o
lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_4/strided_slice_3StridedSlice2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_4/strided_slice_3/stack:output:0'lstm_4/strided_slice_3/stack_1:output:0'lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskl
lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_4/transpose_1	Transpose2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_4/dropout/MulMullstm_4/transpose_1:y:0 dropout_4/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout_4/dropout/ShapeShapelstm_4/transpose_1:y:0*
T0*
_output_shapes
:½
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed*
seed2e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=È
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
lstm_5/ShapeShapedropout_4/dropout/Mul_1:z:0*
T0*
_output_shapes
:d
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_5/transpose	Transposedropout_4/dropout/Mul_1:z:0lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:f
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   õ
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(lstm_5/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¨
lstm_5/lstm_cell_8/MatMulMatMullstm_5/strided_slice_2:output:00lstm_5/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_5/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0¢
lstm_5/lstm_cell_8/MatMul_1MatMullstm_5/zeros:output:02lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/lstm_cell_8/addAddV2#lstm_5/lstm_cell_8/MatMul:product:0%lstm_5/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)lstm_5/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp2lstm_5_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
lstm_5/lstm_cell_8/BiasAddBiasAddlstm_5/lstm_cell_8/add:z:01lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"lstm_5/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
lstm_5/lstm_cell_8/splitSplit+lstm_5/lstm_cell_8/split/split_dim:output:0#lstm_5/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
lstm_5/lstm_cell_8/SigmoidSigmoid!lstm_5/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_5/lstm_cell_8/Sigmoid_1Sigmoid!lstm_5/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/lstm_cell_8/mulMul lstm_5/lstm_cell_8/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm_5/lstm_cell_8/ReluRelu!lstm_5/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/lstm_cell_8/mul_1Mullstm_5/lstm_cell_8/Sigmoid:y:0%lstm_5/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/lstm_cell_8/add_1AddV2lstm_5/lstm_cell_8/mul:z:0lstm_5/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_5/lstm_cell_8/Sigmoid_2Sigmoid!lstm_5/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_5/lstm_cell_8/Relu_1Relulstm_5/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/lstm_cell_8/mul_2Mul lstm_5/lstm_cell_8/Sigmoid_2:y:0'lstm_5/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Í
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ß
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_5_lstm_cell_8_matmul_readvariableop_resource3lstm_5_lstm_cell_8_matmul_1_readvariableop_resource2lstm_5_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_5_while_body_37270*#
condR
lstm_5_while_cond_37269*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0o
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskl
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_5/dropout/MulMullstm_5/strided_slice_3:output:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dropout_5/dropout/ShapeShapelstm_5/strided_slice_3:output:0*
T0*
_output_shapes
:¹
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed*
seed2e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ä
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_1/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp)^lstm_3/lstm_cell_6/MatMul/ReadVariableOp+^lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp^lstm_3/while*^lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp)^lstm_4/lstm_cell_7/MatMul/ReadVariableOp+^lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp^lstm_4/while*^lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp)^lstm_5/lstm_cell_8/MatMul/ReadVariableOp+^lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp^lstm_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp)lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp2T
(lstm_3/lstm_cell_6/MatMul/ReadVariableOp(lstm_3/lstm_cell_6/MatMul/ReadVariableOp2X
*lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp*lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp2
lstm_3/whilelstm_3/while2V
)lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp)lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp2T
(lstm_4/lstm_cell_7/MatMul/ReadVariableOp(lstm_4/lstm_cell_7/MatMul/ReadVariableOp2X
*lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp*lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp2
lstm_4/whilelstm_4/while2V
)lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp)lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp2T
(lstm_5/lstm_cell_8/MatMul/ReadVariableOp(lstm_5/lstm_cell_8/MatMul/ReadVariableOp2X
*lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp*lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp2
lstm_5/whilelstm_5/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
²
&__inference_lstm_4_layer_call_fn_38051
inputs_0
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_34603|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Í7
À
while_body_38143
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_7_matmul_readvariableop_resource_0:F
4while_lstm_cell_7_matmul_1_readvariableop_resource_0:A
3while_lstm_cell_7_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_7_matmul_readvariableop_resource:D
2while_lstm_cell_7_matmul_1_readvariableop_resource:?
1while_lstm_cell_7_biasadd_readvariableop_resource:¢(while/lstm_cell_7/BiasAdd/ReadVariableOp¢'while/lstm_cell_7/MatMul/ReadVariableOp¢)while/lstm_cell_7/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0·
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0£
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_7/BiasAdd/ReadVariableOp(^while/lstm_cell_7/MatMul/ReadVariableOp*^while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_7/BiasAdd/ReadVariableOp(while/lstm_cell_7/BiasAdd/ReadVariableOp2R
'while/lstm_cell_7/MatMul/ReadVariableOp'while/lstm_cell_7/MatMul/ReadVariableOp2V
)while/lstm_cell_7/MatMul_1/ReadVariableOp)while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°
¾
while_cond_38785
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_38785___redundant_placeholder03
/while_while_cond_38785___redundant_placeholder13
/while_while_cond_38785___redundant_placeholder23
/while_while_cond_38785___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
O
À
$sequential_1_lstm_3_while_body_33732D
@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counterJ
Fsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations)
%sequential_1_lstm_3_while_placeholder+
'sequential_1_lstm_3_while_placeholder_1+
'sequential_1_lstm_3_while_placeholder_2+
'sequential_1_lstm_3_while_placeholder_3C
?sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1_0
{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0X
Fsequential_1_lstm_3_while_lstm_cell_6_matmul_readvariableop_resource_0:,Z
Hsequential_1_lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource_0:,U
Gsequential_1_lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource_0:,&
"sequential_1_lstm_3_while_identity(
$sequential_1_lstm_3_while_identity_1(
$sequential_1_lstm_3_while_identity_2(
$sequential_1_lstm_3_while_identity_3(
$sequential_1_lstm_3_while_identity_4(
$sequential_1_lstm_3_while_identity_5A
=sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1}
ysequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensorV
Dsequential_1_lstm_3_while_lstm_cell_6_matmul_readvariableop_resource:,X
Fsequential_1_lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource:,S
Esequential_1_lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource:,¢<sequential_1/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp¢;sequential_1/lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp¢=sequential_1/lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp
Ksequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
=sequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_3_while_placeholderTsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Â
;sequential_1/lstm_3/while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOpFsequential_1_lstm_3_while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:,*
dtype0ó
,sequential_1/lstm_3/while/lstm_cell_6/MatMulMatMulDsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_1/lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,Æ
=sequential_1/lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOpHsequential_1_lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:,*
dtype0Ú
.sequential_1/lstm_3/while/lstm_cell_6/MatMul_1MatMul'sequential_1_lstm_3_while_placeholder_2Esequential_1/lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,Ö
)sequential_1/lstm_3/while/lstm_cell_6/addAddV26sequential_1/lstm_3/while/lstm_cell_6/MatMul:product:08sequential_1/lstm_3/while/lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,À
<sequential_1/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOpGsequential_1_lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:,*
dtype0ß
-sequential_1/lstm_3/while/lstm_cell_6/BiasAddBiasAdd-sequential_1/lstm_3/while/lstm_cell_6/add:z:0Dsequential_1/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,w
5sequential_1/lstm_3/while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¨
+sequential_1/lstm_3/while/lstm_cell_6/splitSplit>sequential_1/lstm_3/while/lstm_cell_6/split/split_dim:output:06sequential_1/lstm_3/while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split 
-sequential_1/lstm_3/while/lstm_cell_6/SigmoidSigmoid4sequential_1/lstm_3/while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
/sequential_1/lstm_3/while/lstm_cell_6/Sigmoid_1Sigmoid4sequential_1/lstm_3/while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
)sequential_1/lstm_3/while/lstm_cell_6/mulMul3sequential_1/lstm_3/while/lstm_cell_6/Sigmoid_1:y:0'sequential_1_lstm_3_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_1/lstm_3/while/lstm_cell_6/ReluRelu4sequential_1/lstm_3/while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
+sequential_1/lstm_3/while/lstm_cell_6/mul_1Mul1sequential_1/lstm_3/while/lstm_cell_6/Sigmoid:y:08sequential_1/lstm_3/while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
+sequential_1/lstm_3/while/lstm_cell_6/add_1AddV2-sequential_1/lstm_3/while/lstm_cell_6/mul:z:0/sequential_1/lstm_3/while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
/sequential_1/lstm_3/while/lstm_cell_6/Sigmoid_2Sigmoid4sequential_1/lstm_3/while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_1/lstm_3/while/lstm_cell_6/Relu_1Relu/sequential_1/lstm_3/while/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
+sequential_1/lstm_3/while/lstm_cell_6/mul_2Mul3sequential_1/lstm_3/while/lstm_cell_6/Sigmoid_2:y:0:sequential_1/lstm_3/while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>sequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_3_while_placeholder_1%sequential_1_lstm_3_while_placeholder/sequential_1/lstm_3/while/lstm_cell_6/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒa
sequential_1/lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_1/lstm_3/while/addAddV2%sequential_1_lstm_3_while_placeholder(sequential_1/lstm_3/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_1/lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :·
sequential_1/lstm_3/while/add_1AddV2@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counter*sequential_1/lstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_1/lstm_3/while/IdentityIdentity#sequential_1/lstm_3/while/add_1:z:0^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: º
$sequential_1/lstm_3/while/Identity_1IdentityFsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: 
$sequential_1/lstm_3/while/Identity_2Identity!sequential_1/lstm_3/while/add:z:0^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: Õ
$sequential_1/lstm_3/while/Identity_3IdentityNsequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: :éèÒ´
$sequential_1/lstm_3/while/Identity_4Identity/sequential_1/lstm_3/while/lstm_cell_6/mul_2:z:0^sequential_1/lstm_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
$sequential_1/lstm_3/while/Identity_5Identity/sequential_1/lstm_3/while/lstm_cell_6/add_1:z:0^sequential_1/lstm_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_1/lstm_3/while/NoOpNoOp=^sequential_1/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp<^sequential_1/lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp>^sequential_1/lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_1_lstm_3_while_identity+sequential_1/lstm_3/while/Identity:output:0"U
$sequential_1_lstm_3_while_identity_1-sequential_1/lstm_3/while/Identity_1:output:0"U
$sequential_1_lstm_3_while_identity_2-sequential_1/lstm_3/while/Identity_2:output:0"U
$sequential_1_lstm_3_while_identity_3-sequential_1/lstm_3/while/Identity_3:output:0"U
$sequential_1_lstm_3_while_identity_4-sequential_1/lstm_3/while/Identity_4:output:0"U
$sequential_1_lstm_3_while_identity_5-sequential_1/lstm_3/while/Identity_5:output:0"
Esequential_1_lstm_3_while_lstm_cell_6_biasadd_readvariableop_resourceGsequential_1_lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource_0"
Fsequential_1_lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resourceHsequential_1_lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource_0"
Dsequential_1_lstm_3_while_lstm_cell_6_matmul_readvariableop_resourceFsequential_1_lstm_3_while_lstm_cell_6_matmul_readvariableop_resource_0"
=sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1?sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1_0"ø
ysequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2|
<sequential_1/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp<sequential_1/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp2z
;sequential_1/lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp;sequential_1/lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp2~
=sequential_1/lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp=sequential_1/lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ê"
Ï
while_body_34725
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_7_34749_0:+
while_lstm_cell_7_34751_0:'
while_lstm_cell_7_34753_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_7_34749:)
while_lstm_cell_7_34751:%
while_lstm_cell_7_34753:¢)while/lstm_cell_7/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0ª
)while/lstm_cell_7/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_7_34749_0while_lstm_cell_7_34751_0while_lstm_cell_7_34753_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_34666Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_7/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/lstm_cell_7/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity2while/lstm_cell_7/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

while/NoOpNoOp*^while/lstm_cell_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_7_34749while_lstm_cell_7_34749_0"4
while_lstm_cell_7_34751while_lstm_cell_7_34751_0"4
while_lstm_cell_7_34753while_lstm_cell_7_34753_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_7/StatefulPartitionedCall)while/lstm_cell_7/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
J

A__inference_lstm_5_layer_call_and_return_conditional_losses_39013
inputs_0<
*lstm_cell_8_matmul_readvariableop_resource:>
,lstm_cell_8_matmul_1_readvariableop_resource:9
+lstm_cell_8_biasadd_readvariableop_resource:
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_38929*
condR
while_cond_38928*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0


c
D__inference_dropout_3_layer_call_and_return_conditional_losses_36079

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
¾
while_cond_34724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_34724___redundant_placeholder03
/while_while_cond_34724___redundant_placeholder13
/while_while_cond_34724___redundant_placeholder23
/while_while_cond_34724___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¾

'__inference_dense_1_layer_call_fn_39335

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_35641o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
¾
while_cond_39214
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_39214___redundant_placeholder03
/while_while_cond_39214___redundant_placeholder13
/while_while_cond_39214___redundant_placeholder23
/while_while_cond_39214___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ô	
Ê
lstm_4_while_cond_36685*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1A
=lstm_4_while_lstm_4_while_cond_36685___redundant_placeholder0A
=lstm_4_while_lstm_4_while_cond_36685___redundant_placeholder1A
=lstm_4_while_lstm_4_while_cond_36685___redundant_placeholder2A
=lstm_4_while_lstm_4_while_cond_36685___redundant_placeholder3
lstm_4_while_identity
~
lstm_4/while/LessLesslstm_4_while_placeholder(lstm_4_while_less_lstm_4_strided_slice_1*
T0*
_output_shapes
: Y
lstm_4/while/IdentityIdentitylstm_4/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_4_while_identitylstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Í7
À
while_body_37786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_6_matmul_readvariableop_resource_0:,F
4while_lstm_cell_6_matmul_1_readvariableop_resource_0:,A
3while_lstm_cell_6_biasadd_readvariableop_resource_0:,
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_6_matmul_readvariableop_resource:,D
2while_lstm_cell_6_matmul_1_readvariableop_resource:,?
1while_lstm_cell_6_biasadd_readvariableop_resource:,¢(while/lstm_cell_6/BiasAdd/ReadVariableOp¢'while/lstm_cell_6/MatMul/ReadVariableOp¢)while/lstm_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:,*
dtype0·
while/lstm_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
)while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:,*
dtype0
while/lstm_cell_6/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
while/lstm_cell_6/addAddV2"while/lstm_cell_6/MatMul:product:0$while/lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
(while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:,*
dtype0£
while/lstm_cell_6/BiasAddBiasAddwhile/lstm_cell_6/add:z:00while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,c
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0"while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_6/SigmoidSigmoid while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_6/Sigmoid_1Sigmoid while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mulMulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_6/ReluRelu while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mul_1Mulwhile/lstm_cell_6/Sigmoid:y:0$while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/add_1AddV2while/lstm_cell_6/mul:z:0while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_6/Sigmoid_2Sigmoid while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_6/Relu_1Reluwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mul_2Mulwhile/lstm_cell_6/Sigmoid_2:y:0&while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_6/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_6/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_6/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_6_biasadd_readvariableop_resource3while_lstm_cell_6_biasadd_readvariableop_resource_0"j
2while_lstm_cell_6_matmul_1_readvariableop_resource4while_lstm_cell_6_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_6_matmul_readvariableop_resource2while_lstm_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_6/BiasAdd/ReadVariableOp(while/lstm_cell_6/BiasAdd/ReadVariableOp2R
'while/lstm_cell_6/MatMul/ReadVariableOp'while/lstm_cell_6/MatMul/ReadVariableOp2V
)while/lstm_cell_6/MatMul_1/ReadVariableOp)while/lstm_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
J

A__inference_lstm_4_layer_call_and_return_conditional_losses_38370
inputs_0<
*lstm_cell_7_matmul_readvariableop_resource:>
,lstm_cell_7_matmul_1_readvariableop_resource:9
+lstm_cell_7_biasadd_readvariableop_resource:
identity¢"lstm_cell_7/BiasAdd/ReadVariableOp¢!lstm_cell_7/MatMul/ReadVariableOp¢#lstm_cell_7/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_38286*
condR
while_cond_38285*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_7/BiasAdd/ReadVariableOp"^lstm_cell_7/MatMul/ReadVariableOp$^lstm_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_7/BiasAdd/ReadVariableOp"lstm_cell_7/BiasAdd/ReadVariableOp2F
!lstm_cell_7/MatMul/ReadVariableOp!lstm_cell_7/MatMul/ReadVariableOp2J
#lstm_cell_7/MatMul_1/ReadVariableOp#lstm_cell_7/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
°
¾
while_cond_38285
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_38285___redundant_placeholder03
/while_while_cond_38285___redundant_placeholder13
/while_while_cond_38285___redundant_placeholder23
/while_while_cond_38285___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ò
Ã
G__inference_sequential_1_layer_call_and_return_conditional_losses_36394
lstm_3_input
lstm_3_36364:,
lstm_3_36366:,
lstm_3_36368:,
lstm_4_36372:
lstm_4_36374:
lstm_4_36376:
lstm_5_36380:
lstm_5_36382:
lstm_5_36384:
dense_1_36388:
dense_1_36390:
identity¢dense_1/StatefulPartitionedCall¢lstm_3/StatefulPartitionedCall¢lstm_4/StatefulPartitionedCall¢lstm_5/StatefulPartitionedCallÿ
lstm_3/StatefulPartitionedCallStatefulPartitionedCalllstm_3_inputlstm_3_36364lstm_3_36366lstm_3_36368*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_35302Þ
dropout_3/PartitionedCallPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_35315
lstm_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0lstm_4_36372lstm_4_36374lstm_4_36376*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_35459Þ
dropout_4/PartitionedCallPartitionedCall'lstm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_35472
lstm_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0lstm_5_36380lstm_5_36382lstm_5_36384*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_35616Ú
dropout_5/PartitionedCallPartitionedCall'lstm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_35629
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_1_36388dense_1_36390*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_35641w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
NoOpNoOp ^dense_1/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_3_input
$
¯
G__inference_sequential_1_layer_call_and_return_conditional_losses_36427
lstm_3_input
lstm_3_36397:,
lstm_3_36399:,
lstm_3_36401:,
lstm_4_36405:
lstm_4_36407:
lstm_4_36409:
lstm_5_36413:
lstm_5_36415:
lstm_5_36417:
dense_1_36421:
dense_1_36423:
identity¢dense_1/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢lstm_3/StatefulPartitionedCall¢lstm_4/StatefulPartitionedCall¢lstm_5/StatefulPartitionedCallÿ
lstm_3/StatefulPartitionedCallStatefulPartitionedCalllstm_3_inputlstm_3_36397lstm_3_36399lstm_3_36401*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_36238î
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_36079
lstm_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0lstm_4_36405lstm_4_36407lstm_4_36409*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_36050
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_35891
lstm_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0lstm_5_36413lstm_5_36415lstm_5_36417*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_35862
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_35703
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_1_36421dense_1_36423*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_35641w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp ^dense_1/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_3_input
á
Î
$sequential_1_lstm_4_while_cond_33871D
@sequential_1_lstm_4_while_sequential_1_lstm_4_while_loop_counterJ
Fsequential_1_lstm_4_while_sequential_1_lstm_4_while_maximum_iterations)
%sequential_1_lstm_4_while_placeholder+
'sequential_1_lstm_4_while_placeholder_1+
'sequential_1_lstm_4_while_placeholder_2+
'sequential_1_lstm_4_while_placeholder_3F
Bsequential_1_lstm_4_while_less_sequential_1_lstm_4_strided_slice_1[
Wsequential_1_lstm_4_while_sequential_1_lstm_4_while_cond_33871___redundant_placeholder0[
Wsequential_1_lstm_4_while_sequential_1_lstm_4_while_cond_33871___redundant_placeholder1[
Wsequential_1_lstm_4_while_sequential_1_lstm_4_while_cond_33871___redundant_placeholder2[
Wsequential_1_lstm_4_while_sequential_1_lstm_4_while_cond_33871___redundant_placeholder3&
"sequential_1_lstm_4_while_identity
²
sequential_1/lstm_4/while/LessLess%sequential_1_lstm_4_while_placeholderBsequential_1_lstm_4_while_less_sequential_1_lstm_4_strided_slice_1*
T0*
_output_shapes
: s
"sequential_1/lstm_4/while/IdentityIdentity"sequential_1/lstm_4/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_1_lstm_4_while_identity+sequential_1/lstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ð¢

!__inference__traced_restore_39912
file_prefix1
assignvariableop_dense_1_kernel:-
assignvariableop_1_dense_1_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: >
,assignvariableop_7_lstm_3_lstm_cell_6_kernel:,H
6assignvariableop_8_lstm_3_lstm_cell_6_recurrent_kernel:,8
*assignvariableop_9_lstm_3_lstm_cell_6_bias:,?
-assignvariableop_10_lstm_4_lstm_cell_7_kernel:I
7assignvariableop_11_lstm_4_lstm_cell_7_recurrent_kernel:9
+assignvariableop_12_lstm_4_lstm_cell_7_bias:?
-assignvariableop_13_lstm_5_lstm_cell_8_kernel:I
7assignvariableop_14_lstm_5_lstm_cell_8_recurrent_kernel:9
+assignvariableop_15_lstm_5_lstm_cell_8_bias:#
assignvariableop_16_total: #
assignvariableop_17_count: ;
)assignvariableop_18_adam_dense_1_kernel_m:5
'assignvariableop_19_adam_dense_1_bias_m:F
4assignvariableop_20_adam_lstm_3_lstm_cell_6_kernel_m:,P
>assignvariableop_21_adam_lstm_3_lstm_cell_6_recurrent_kernel_m:,@
2assignvariableop_22_adam_lstm_3_lstm_cell_6_bias_m:,F
4assignvariableop_23_adam_lstm_4_lstm_cell_7_kernel_m:P
>assignvariableop_24_adam_lstm_4_lstm_cell_7_recurrent_kernel_m:@
2assignvariableop_25_adam_lstm_4_lstm_cell_7_bias_m:F
4assignvariableop_26_adam_lstm_5_lstm_cell_8_kernel_m:P
>assignvariableop_27_adam_lstm_5_lstm_cell_8_recurrent_kernel_m:@
2assignvariableop_28_adam_lstm_5_lstm_cell_8_bias_m:;
)assignvariableop_29_adam_dense_1_kernel_v:5
'assignvariableop_30_adam_dense_1_bias_v:F
4assignvariableop_31_adam_lstm_3_lstm_cell_6_kernel_v:,P
>assignvariableop_32_adam_lstm_3_lstm_cell_6_recurrent_kernel_v:,@
2assignvariableop_33_adam_lstm_3_lstm_cell_6_bias_v:,F
4assignvariableop_34_adam_lstm_4_lstm_cell_7_kernel_v:P
>assignvariableop_35_adam_lstm_4_lstm_cell_7_recurrent_kernel_v:@
2assignvariableop_36_adam_lstm_4_lstm_cell_7_bias_v:F
4assignvariableop_37_adam_lstm_5_lstm_cell_8_kernel_v:P
>assignvariableop_38_adam_lstm_5_lstm_cell_8_recurrent_kernel_v:@
2assignvariableop_39_adam_lstm_5_lstm_cell_8_bias_v:
identity_41¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9È
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*î
valueäBá)B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÂ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B î
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_3_lstm_cell_6_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_8AssignVariableOp6assignvariableop_8_lstm_3_lstm_cell_6_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp*assignvariableop_9_lstm_3_lstm_cell_6_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp-assignvariableop_10_lstm_4_lstm_cell_7_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_11AssignVariableOp7assignvariableop_11_lstm_4_lstm_cell_7_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp+assignvariableop_12_lstm_4_lstm_cell_7_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp-assignvariableop_13_lstm_5_lstm_cell_8_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_14AssignVariableOp7assignvariableop_14_lstm_5_lstm_cell_8_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp+assignvariableop_15_lstm_5_lstm_cell_8_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_1_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_1_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_lstm_3_lstm_cell_6_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_21AssignVariableOp>assignvariableop_21_adam_lstm_3_lstm_cell_6_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_lstm_3_lstm_cell_6_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_lstm_4_lstm_cell_7_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_24AssignVariableOp>assignvariableop_24_adam_lstm_4_lstm_cell_7_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_lstm_4_lstm_cell_7_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_lstm_5_lstm_cell_8_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_27AssignVariableOp>assignvariableop_27_adam_lstm_5_lstm_cell_8_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_28AssignVariableOp2assignvariableop_28_adam_lstm_5_lstm_cell_8_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_lstm_3_lstm_cell_6_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_32AssignVariableOp>assignvariableop_32_adam_lstm_3_lstm_cell_6_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_33AssignVariableOp2assignvariableop_33_adam_lstm_3_lstm_cell_6_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_lstm_4_lstm_cell_7_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_35AssignVariableOp>assignvariableop_35_adam_lstm_4_lstm_cell_7_recurrent_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_36AssignVariableOp2assignvariableop_36_adam_lstm_4_lstm_cell_7_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_lstm_5_lstm_cell_8_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_38AssignVariableOp>assignvariableop_38_adam_lstm_5_lstm_cell_8_recurrent_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_39AssignVariableOp2assignvariableop_39_adam_lstm_5_lstm_cell_8_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¿
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_41IdentityIdentity_40:output:0^NoOp_1*
T0*
_output_shapes
: ¬
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_41Identity_41:output:0*e
_input_shapesT
R: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ê"
Ï
while_body_34884
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_8_34908_0:+
while_lstm_cell_8_34910_0:'
while_lstm_cell_8_34912_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_8_34908:)
while_lstm_cell_8_34910:%
while_lstm_cell_8_34912:¢)while/lstm_cell_8/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0ª
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_34908_0while_lstm_cell_8_34910_0while_lstm_cell_8_34912_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_34870Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

while/NoOpNoOp*^while/lstm_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_8_34908while_lstm_cell_8_34908_0"4
while_lstm_cell_8_34910while_lstm_cell_8_34910_0"4
while_lstm_cell_8_34912while_lstm_cell_8_34912_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ê"
Ï
while_body_35075
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_8_35099_0:+
while_lstm_cell_8_35101_0:'
while_lstm_cell_8_35103_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_8_35099:)
while_lstm_cell_8_35101:%
while_lstm_cell_8_35103:¢)while/lstm_cell_8/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0ª
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_35099_0while_lstm_cell_8_35101_0while_lstm_cell_8_35103_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_35016Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

while/NoOpNoOp*^while/lstm_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_8_35099while_lstm_cell_8_35099_0"4
while_lstm_cell_8_35101while_lstm_cell_8_35101_0"4
while_lstm_cell_8_35103while_lstm_cell_8_35103_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ï?
 

lstm_4_while_body_36686*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0K
9lstm_4_while_lstm_cell_7_matmul_readvariableop_resource_0:M
;lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource_0:H
:lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource_0:
lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensorI
7lstm_4_while_lstm_cell_7_matmul_readvariableop_resource:K
9lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource:F
8lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource:¢/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp¢.lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp¢0lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   É
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.lstm_4/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp9lstm_4_while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ì
lstm_4/while/lstm_cell_7/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
0lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp;lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0³
!lstm_4/while/lstm_cell_7/MatMul_1MatMullstm_4_while_placeholder_28lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
lstm_4/while/lstm_cell_7/addAddV2)lstm_4/while/lstm_cell_7/MatMul:product:0+lstm_4/while/lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0¸
 lstm_4/while/lstm_cell_7/BiasAddBiasAdd lstm_4/while/lstm_cell_7/add:z:07lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(lstm_4/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_4/while/lstm_cell_7/splitSplit1lstm_4/while/lstm_cell_7/split/split_dim:output:0)lstm_4/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 lstm_4/while/lstm_cell_7/SigmoidSigmoid'lstm_4/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_4/while/lstm_cell_7/Sigmoid_1Sigmoid'lstm_4/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/while/lstm_cell_7/mulMul&lstm_4/while/lstm_cell_7/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/while/lstm_cell_7/ReluRelu'lstm_4/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
lstm_4/while/lstm_cell_7/mul_1Mul$lstm_4/while/lstm_cell_7/Sigmoid:y:0+lstm_4/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/while/lstm_cell_7/add_1AddV2 lstm_4/while/lstm_cell_7/mul:z:0"lstm_4/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_4/while/lstm_cell_7/Sigmoid_2Sigmoid'lstm_4/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_4/while/lstm_cell_7/Relu_1Relu"lstm_4/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
lstm_4/while/lstm_cell_7/mul_2Mul&lstm_4/while/lstm_cell_7/Sigmoid_2:y:0-lstm_4/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
1lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_4_while_placeholder_1lstm_4_while_placeholder"lstm_4/while/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_4/while/addAddV2lstm_4_while_placeholderlstm_4/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_4/while/add_1AddV2&lstm_4_while_lstm_4_while_loop_counterlstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_4/while/IdentityIdentitylstm_4/while/add_1:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 
lstm_4/while/Identity_1Identity,lstm_4_while_lstm_4_while_maximum_iterations^lstm_4/while/NoOp*
T0*
_output_shapes
: n
lstm_4/while/Identity_2Identitylstm_4/while/add:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: ®
lstm_4/while/Identity_3IdentityAlstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_4/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_4/while/Identity_4Identity"lstm_4/while/lstm_cell_7/mul_2:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_7/add_1:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
lstm_4/while/NoOpNoOp0^lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp/^lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp1^lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_4_while_identitylstm_4/while/Identity:output:0";
lstm_4_while_identity_1 lstm_4/while/Identity_1:output:0";
lstm_4_while_identity_2 lstm_4/while/Identity_2:output:0";
lstm_4_while_identity_3 lstm_4/while/Identity_3:output:0";
lstm_4_while_identity_4 lstm_4/while/Identity_4:output:0";
lstm_4_while_identity_5 lstm_4/while/Identity_5:output:0"L
#lstm_4_while_lstm_4_strided_slice_1%lstm_4_while_lstm_4_strided_slice_1_0"v
8lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource:lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource_0"x
9lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource;lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource_0"t
7lstm_4_while_lstm_cell_7_matmul_readvariableop_resource9lstm_4_while_lstm_cell_7_matmul_readvariableop_resource_0"Ä
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensoralstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2b
/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp2`
.lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp.lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp2d
0lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp0lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ï?
 

lstm_3_while_body_36546*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0K
9lstm_3_while_lstm_cell_6_matmul_readvariableop_resource_0:,M
;lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource_0:,H
:lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource_0:,
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorI
7lstm_3_while_lstm_cell_6_matmul_readvariableop_resource:,K
9lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource:,F
8lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource:,¢/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp¢.lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp¢0lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   É
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.lstm_3/while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp9lstm_3_while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:,*
dtype0Ì
lstm_3/while/lstm_cell_6/MatMulMatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,¬
0lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp;lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:,*
dtype0³
!lstm_3/while/lstm_cell_6/MatMul_1MatMullstm_3_while_placeholder_28lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,¯
lstm_3/while/lstm_cell_6/addAddV2)lstm_3/while/lstm_cell_6/MatMul:product:0+lstm_3/while/lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,¦
/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:,*
dtype0¸
 lstm_3/while/lstm_cell_6/BiasAddBiasAdd lstm_3/while/lstm_cell_6/add:z:07lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,j
(lstm_3/while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_3/while/lstm_cell_6/splitSplit1lstm_3/while/lstm_cell_6/split/split_dim:output:0)lstm_3/while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 lstm_3/while/lstm_cell_6/SigmoidSigmoid'lstm_3/while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_3/while/lstm_cell_6/Sigmoid_1Sigmoid'lstm_3/while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/while/lstm_cell_6/mulMul&lstm_3/while/lstm_cell_6/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/while/lstm_cell_6/ReluRelu'lstm_3/while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
lstm_3/while/lstm_cell_6/mul_1Mul$lstm_3/while/lstm_cell_6/Sigmoid:y:0+lstm_3/while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/while/lstm_cell_6/add_1AddV2 lstm_3/while/lstm_cell_6/mul:z:0"lstm_3/while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_3/while/lstm_cell_6/Sigmoid_2Sigmoid'lstm_3/while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_3/while/lstm_cell_6/Relu_1Relu"lstm_3/while/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
lstm_3/while/lstm_cell_6/mul_2Mul&lstm_3/while/lstm_cell_6/Sigmoid_2:y:0-lstm_3/while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder"lstm_3/while/lstm_cell_6/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations^lstm_3/while/NoOp*
T0*
_output_shapes
: n
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: ®
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_3/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_3/while/Identity_4Identity"lstm_3/while/lstm_cell_6/mul_2:z:0^lstm_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_6/add_1:z:0^lstm_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
lstm_3/while/NoOpNoOp0^lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"v
8lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource:lstm_3_while_lstm_cell_6_biasadd_readvariableop_resource_0"x
9lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource;lstm_3_while_lstm_cell_6_matmul_1_readvariableop_resource_0"t
7lstm_3_while_lstm_cell_6_matmul_readvariableop_resource9lstm_3_while_lstm_cell_6_matmul_readvariableop_resource_0"Ä
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2b
/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp/lstm_3/while/lstm_cell_6/BiasAdd/ReadVariableOp2`
.lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp.lstm_3/while/lstm_cell_6/MatMul/ReadVariableOp2d
0lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp0lstm_3/while/lstm_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°
¾
while_cond_36153
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_36153___redundant_placeholder03
/while_while_cond_36153___redundant_placeholder13
/while_while_cond_36153___redundant_placeholder23
/while_while_cond_36153___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
É
ÿ
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_35016

inputs

states
states_10
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
°
¾
while_cond_35965
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_35965___redundant_placeholder03
/while_while_cond_35965___redundant_placeholder13
/while_while_cond_35965___redundant_placeholder23
/while_while_cond_35965___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
°
¾
while_cond_38428
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_38428___redundant_placeholder03
/while_while_cond_38428___redundant_placeholder13
/while_while_cond_38428___redundant_placeholder23
/while_while_cond_38428___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ñ

F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_39411

inputs
states_0
states_10
matmul_readvariableop_resource:,2
 matmul_1_readvariableop_resource:,-
biasadd_readvariableop_resource:,
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:,*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ÚI

A__inference_lstm_3_layer_call_and_return_conditional_losses_38013

inputs<
*lstm_cell_6_matmul_readvariableop_resource:,>
,lstm_cell_6_matmul_1_readvariableop_resource:,9
+lstm_cell_6_biasadd_readvariableop_resource:,
identity¢"lstm_cell_6/BiasAdd/ReadVariableOp¢!lstm_cell_6/MatMul/ReadVariableOp¢#lstm_cell_6/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_6/MatMul/ReadVariableOpReadVariableOp*lstm_cell_6_matmul_readvariableop_resource*
_output_shapes

:,*
dtype0
lstm_cell_6/MatMulMatMulstrided_slice_2:output:0)lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
#lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0
lstm_cell_6/MatMul_1MatMulzeros:output:0+lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
lstm_cell_6/addAddV2lstm_cell_6/MatMul:product:0lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
"lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0
lstm_cell_6/BiasAddBiasAddlstm_cell_6/add:z:0*lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,]
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_6/SigmoidSigmoidlstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_6/mulMullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_6/ReluRelulstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_6/mul_1Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_6/add_1AddV2lstm_cell_6/mul:z:0lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_6/Relu_1Relulstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_6/mul_2Mullstm_cell_6/Sigmoid_2:y:0 lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_6_matmul_readvariableop_resource,lstm_cell_6_matmul_1_readvariableop_resource+lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_37929*
condR
while_cond_37928*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_6/BiasAdd/ReadVariableOp"^lstm_cell_6/MatMul/ReadVariableOp$^lstm_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_6/BiasAdd/ReadVariableOp"lstm_cell_6/BiasAdd/ReadVariableOp2F
!lstm_cell_6/MatMul/ReadVariableOp!lstm_cell_6/MatMul/ReadVariableOp2J
#lstm_cell_6/MatMul_1/ReadVariableOp#lstm_cell_6/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_39443

inputs
states_0
states_10
matmul_readvariableop_resource:,2
 matmul_1_readvariableop_resource:,-
biasadd_readvariableop_resource:,
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:,*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ò7
ù
A__inference_lstm_4_layer_call_and_return_conditional_losses_34794

inputs#
lstm_cell_7_34712:#
lstm_cell_7_34714:
lstm_cell_7_34716:
identity¢#lstm_cell_7/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskì
#lstm_cell_7/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_7_34712lstm_cell_7_34714lstm_cell_7_34716*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_34666n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¯
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_7_34712lstm_cell_7_34714lstm_cell_7_34716*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_34725*
condR
while_cond_34724*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿt
NoOpNoOp$^lstm_cell_7/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_7/StatefulPartitionedCall#lstm_cell_7/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
ÿ
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_34870

inputs

states
states_10
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¢
²
&__inference_lstm_4_layer_call_fn_38062
inputs_0
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_34794|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Å	
ó
B__inference_dense_1_layer_call_and_return_conditional_losses_39345

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
b
)__inference_dropout_4_layer_call_fn_38666

inputs
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_35891s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô	
Ê
lstm_3_while_cond_36545*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1A
=lstm_3_while_lstm_3_while_cond_36545___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_36545___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_36545___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_36545___redundant_placeholder3
lstm_3_while_identity
~
lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: Y
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_3_while_identitylstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÚI

A__inference_lstm_4_layer_call_and_return_conditional_losses_35459

inputs<
*lstm_cell_7_matmul_readvariableop_resource:>
,lstm_cell_7_matmul_1_readvariableop_resource:9
+lstm_cell_7_biasadd_readvariableop_resource:
identity¢"lstm_cell_7/BiasAdd/ReadVariableOp¢!lstm_cell_7/MatMul/ReadVariableOp¢#lstm_cell_7/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35375*
condR
while_cond_35374*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_7/BiasAdd/ReadVariableOp"^lstm_cell_7/MatMul/ReadVariableOp$^lstm_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_7/BiasAdd/ReadVariableOp"lstm_cell_7/BiasAdd/ReadVariableOp2F
!lstm_cell_7/MatMul/ReadVariableOp!lstm_cell_7/MatMul/ReadVariableOp2J
#lstm_cell_7/MatMul_1/ReadVariableOp#lstm_cell_7/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í7
À
while_body_35375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_7_matmul_readvariableop_resource_0:F
4while_lstm_cell_7_matmul_1_readvariableop_resource_0:A
3while_lstm_cell_7_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_7_matmul_readvariableop_resource:D
2while_lstm_cell_7_matmul_1_readvariableop_resource:?
1while_lstm_cell_7_biasadd_readvariableop_resource:¢(while/lstm_cell_7/BiasAdd/ReadVariableOp¢'while/lstm_cell_7/MatMul/ReadVariableOp¢)while/lstm_cell_7/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0·
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0£
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_7/BiasAdd/ReadVariableOp(^while/lstm_cell_7/MatMul/ReadVariableOp*^while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_7/BiasAdd/ReadVariableOp(while/lstm_cell_7/BiasAdd/ReadVariableOp2R
'while/lstm_cell_7/MatMul/ReadVariableOp'while/lstm_cell_7/MatMul/ReadVariableOp2V
)while/lstm_cell_7/MatMul_1/ReadVariableOp)while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
J

A__inference_lstm_3_layer_call_and_return_conditional_losses_37727
inputs_0<
*lstm_cell_6_matmul_readvariableop_resource:,>
,lstm_cell_6_matmul_1_readvariableop_resource:,9
+lstm_cell_6_biasadd_readvariableop_resource:,
identity¢"lstm_cell_6/BiasAdd/ReadVariableOp¢!lstm_cell_6/MatMul/ReadVariableOp¢#lstm_cell_6/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_6/MatMul/ReadVariableOpReadVariableOp*lstm_cell_6_matmul_readvariableop_resource*
_output_shapes

:,*
dtype0
lstm_cell_6/MatMulMatMulstrided_slice_2:output:0)lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
#lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0
lstm_cell_6/MatMul_1MatMulzeros:output:0+lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
lstm_cell_6/addAddV2lstm_cell_6/MatMul:product:0lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
"lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0
lstm_cell_6/BiasAddBiasAddlstm_cell_6/add:z:0*lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,]
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_6/SigmoidSigmoidlstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_6/mulMullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_6/ReluRelulstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_6/mul_1Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_6/add_1AddV2lstm_cell_6/mul:z:0lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_6/Relu_1Relulstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_6/mul_2Mullstm_cell_6/Sigmoid_2:y:0 lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_6_matmul_readvariableop_resource,lstm_cell_6_matmul_1_readvariableop_resource+lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_37643*
condR
while_cond_37642*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_6/BiasAdd/ReadVariableOp"^lstm_cell_6/MatMul/ReadVariableOp$^lstm_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_6/BiasAdd/ReadVariableOp"lstm_cell_6/BiasAdd/ReadVariableOp2F
!lstm_cell_6/MatMul/ReadVariableOp!lstm_cell_6/MatMul/ReadVariableOp2J
#lstm_cell_6/MatMul_1/ReadVariableOp#lstm_cell_6/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ßI

A__inference_lstm_5_layer_call_and_return_conditional_losses_39156

inputs<
*lstm_cell_8_matmul_readvariableop_resource:>
,lstm_cell_8_matmul_1_readvariableop_resource:9
+lstm_cell_8_biasadd_readvariableop_resource:
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39072*
condR
while_cond_39071*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
J

A__inference_lstm_4_layer_call_and_return_conditional_losses_38227
inputs_0<
*lstm_cell_7_matmul_readvariableop_resource:>
,lstm_cell_7_matmul_1_readvariableop_resource:9
+lstm_cell_7_biasadd_readvariableop_resource:
identity¢"lstm_cell_7/BiasAdd/ReadVariableOp¢!lstm_cell_7/MatMul/ReadVariableOp¢#lstm_cell_7/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_38143*
condR
while_cond_38142*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_7/BiasAdd/ReadVariableOp"^lstm_cell_7/MatMul/ReadVariableOp$^lstm_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_7/BiasAdd/ReadVariableOp"lstm_cell_7/BiasAdd/ReadVariableOp2F
!lstm_cell_7/MatMul/ReadVariableOp!lstm_cell_7/MatMul/ReadVariableOp2J
#lstm_cell_7/MatMul_1/ReadVariableOp#lstm_cell_7/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÚI

A__inference_lstm_4_layer_call_and_return_conditional_losses_38656

inputs<
*lstm_cell_7_matmul_readvariableop_resource:>
,lstm_cell_7_matmul_1_readvariableop_resource:9
+lstm_cell_7_biasadd_readvariableop_resource:
identity¢"lstm_cell_7/BiasAdd/ReadVariableOp¢!lstm_cell_7/MatMul/ReadVariableOp¢#lstm_cell_7/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_38572*
condR
while_cond_38571*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_7/BiasAdd/ReadVariableOp"^lstm_cell_7/MatMul/ReadVariableOp$^lstm_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_7/BiasAdd/ReadVariableOp"lstm_cell_7/BiasAdd/ReadVariableOp2F
!lstm_cell_7/MatMul/ReadVariableOp!lstm_cell_7/MatMul/ReadVariableOp2J
#lstm_cell_7/MatMul_1/ReadVariableOp#lstm_cell_7/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_39607

inputs
states_0
states_10
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
×
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_35629

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
ÿ
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_34316

inputs

states
states_10
matmul_readvariableop_resource:,2
 matmul_1_readvariableop_resource:,-
biasadd_readvariableop_resource:,
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:,*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:,*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ñ

F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_39509

inputs
states_0
states_10
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1


c
D__inference_dropout_3_layer_call_and_return_conditional_losses_38040

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í7
À
while_body_36154
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_6_matmul_readvariableop_resource_0:,F
4while_lstm_cell_6_matmul_1_readvariableop_resource_0:,A
3while_lstm_cell_6_biasadd_readvariableop_resource_0:,
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_6_matmul_readvariableop_resource:,D
2while_lstm_cell_6_matmul_1_readvariableop_resource:,?
1while_lstm_cell_6_biasadd_readvariableop_resource:,¢(while/lstm_cell_6/BiasAdd/ReadVariableOp¢'while/lstm_cell_6/MatMul/ReadVariableOp¢)while/lstm_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:,*
dtype0·
while/lstm_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
)while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:,*
dtype0
while/lstm_cell_6/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
while/lstm_cell_6/addAddV2"while/lstm_cell_6/MatMul:product:0$while/lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
(while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:,*
dtype0£
while/lstm_cell_6/BiasAddBiasAddwhile/lstm_cell_6/add:z:00while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,c
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0"while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_6/SigmoidSigmoid while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_6/Sigmoid_1Sigmoid while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mulMulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_6/ReluRelu while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mul_1Mulwhile/lstm_cell_6/Sigmoid:y:0$while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/add_1AddV2while/lstm_cell_6/mul:z:0while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_6/Sigmoid_2Sigmoid while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_6/Relu_1Reluwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mul_2Mulwhile/lstm_cell_6/Sigmoid_2:y:0&while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_6/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_6/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_6/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_6_biasadd_readvariableop_resource3while_lstm_cell_6_biasadd_readvariableop_resource_0"j
2while_lstm_cell_6_matmul_1_readvariableop_resource4while_lstm_cell_6_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_6_matmul_readvariableop_resource2while_lstm_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_6/BiasAdd/ReadVariableOp(while/lstm_cell_6/BiasAdd/ReadVariableOp2R
'while/lstm_cell_6/MatMul/ReadVariableOp'while/lstm_cell_6/MatMul/ReadVariableOp2V
)while/lstm_cell_6/MatMul_1/ReadVariableOp)while/lstm_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Í7
À
while_body_38572
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_7_matmul_readvariableop_resource_0:F
4while_lstm_cell_7_matmul_1_readvariableop_resource_0:A
3while_lstm_cell_7_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_7_matmul_readvariableop_resource:D
2while_lstm_cell_7_matmul_1_readvariableop_resource:?
1while_lstm_cell_7_biasadd_readvariableop_resource:¢(while/lstm_cell_7/BiasAdd/ReadVariableOp¢'while/lstm_cell_7/MatMul/ReadVariableOp¢)while/lstm_cell_7/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0·
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0£
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_7/BiasAdd/ReadVariableOp(^while/lstm_cell_7/MatMul/ReadVariableOp*^while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_7/BiasAdd/ReadVariableOp(while/lstm_cell_7/BiasAdd/ReadVariableOp2R
'while/lstm_cell_7/MatMul/ReadVariableOp'while/lstm_cell_7/MatMul/ReadVariableOp2V
)while/lstm_cell_7/MatMul_1/ReadVariableOp)while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Í7
À
while_body_35532
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_8_matmul_readvariableop_resource_0:F
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:A
3while_lstm_cell_8_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_8_matmul_readvariableop_resource:D
2while_lstm_cell_8_matmul_1_readvariableop_resource:?
1while_lstm_cell_8_biasadd_readvariableop_resource:¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0·
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0£
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ç
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_35472

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_39639

inputs
states_0
states_10
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
J

A__inference_lstm_5_layer_call_and_return_conditional_losses_38870
inputs_0<
*lstm_cell_8_matmul_readvariableop_resource:>
,lstm_cell_8_matmul_1_readvariableop_resource:9
+lstm_cell_8_biasadd_readvariableop_resource:
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_38786*
condR
while_cond_38785*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
°
¾
while_cond_35074
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_35074___redundant_placeholder03
/while_while_cond_35074___redundant_placeholder13
/while_while_cond_35074___redundant_placeholder23
/while_while_cond_35074___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ØU
«
__inference__traced_save_39782
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_3_lstm_cell_6_kernel_read_readvariableopB
>savev2_lstm_3_lstm_cell_6_recurrent_kernel_read_readvariableop6
2savev2_lstm_3_lstm_cell_6_bias_read_readvariableop8
4savev2_lstm_4_lstm_cell_7_kernel_read_readvariableopB
>savev2_lstm_4_lstm_cell_7_recurrent_kernel_read_readvariableop6
2savev2_lstm_4_lstm_cell_7_bias_read_readvariableop8
4savev2_lstm_5_lstm_cell_8_kernel_read_readvariableopB
>savev2_lstm_5_lstm_cell_8_recurrent_kernel_read_readvariableop6
2savev2_lstm_5_lstm_cell_8_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_6_kernel_m_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_6_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_6_bias_m_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_7_kernel_m_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_7_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_7_bias_m_read_readvariableop?
;savev2_adam_lstm_5_lstm_cell_8_kernel_m_read_readvariableopI
Esavev2_adam_lstm_5_lstm_cell_8_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_5_lstm_cell_8_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_6_kernel_v_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_6_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_6_bias_v_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_7_kernel_v_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_7_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_7_bias_v_read_readvariableop?
;savev2_adam_lstm_5_lstm_cell_8_kernel_v_read_readvariableopI
Esavev2_adam_lstm_5_lstm_cell_8_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_5_lstm_cell_8_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Å
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*î
valueäBá)B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¿
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ø
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_3_lstm_cell_6_kernel_read_readvariableop>savev2_lstm_3_lstm_cell_6_recurrent_kernel_read_readvariableop2savev2_lstm_3_lstm_cell_6_bias_read_readvariableop4savev2_lstm_4_lstm_cell_7_kernel_read_readvariableop>savev2_lstm_4_lstm_cell_7_recurrent_kernel_read_readvariableop2savev2_lstm_4_lstm_cell_7_bias_read_readvariableop4savev2_lstm_5_lstm_cell_8_kernel_read_readvariableop>savev2_lstm_5_lstm_cell_8_recurrent_kernel_read_readvariableop2savev2_lstm_5_lstm_cell_8_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop;savev2_adam_lstm_3_lstm_cell_6_kernel_m_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_6_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_3_lstm_cell_6_bias_m_read_readvariableop;savev2_adam_lstm_4_lstm_cell_7_kernel_m_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_7_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_4_lstm_cell_7_bias_m_read_readvariableop;savev2_adam_lstm_5_lstm_cell_8_kernel_m_read_readvariableopEsavev2_adam_lstm_5_lstm_cell_8_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_5_lstm_cell_8_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop;savev2_adam_lstm_3_lstm_cell_6_kernel_v_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_6_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_3_lstm_cell_6_bias_v_read_readvariableop;savev2_adam_lstm_4_lstm_cell_7_kernel_v_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_7_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_4_lstm_cell_7_bias_v_read_readvariableop;savev2_adam_lstm_5_lstm_cell_8_kernel_v_read_readvariableopEsavev2_adam_lstm_5_lstm_cell_8_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_5_lstm_cell_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Á
_input_shapes¯
¬: ::: : : : : :,:,:,::::::: : :::,:,:,:::::::::,:,:,::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:,:$	 

_output_shapes

:,: 


_output_shapes
:,:$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:,:$ 

_output_shapes

:,: 

_output_shapes
:,:$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:,:$! 

_output_shapes

:,: "

_output_shapes
:,:$# 

_output_shapes

::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

::$' 

_output_shapes

:: (

_output_shapes
::)

_output_shapes
: 
°
¾
while_cond_38142
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_38142___redundant_placeholder03
/while_while_cond_38142___redundant_placeholder13
/while_while_cond_38142___redundant_placeholder23
/while_while_cond_38142___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


c
D__inference_dropout_4_layer_call_and_return_conditional_losses_38683

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
ñ
+__inference_lstm_cell_7_layer_call_fn_39460

inputs
states_0
states_1
unknown:
	unknown_0:
	unknown_1:
identity

identity_1

identity_2¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_34520o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
°
¾
while_cond_39071
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_39071___redundant_placeholder03
/while_while_cond_39071___redundant_placeholder13
/while_while_cond_39071___redundant_placeholder23
/while_while_cond_39071___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
á
Î
$sequential_1_lstm_3_while_cond_33731D
@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counterJ
Fsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations)
%sequential_1_lstm_3_while_placeholder+
'sequential_1_lstm_3_while_placeholder_1+
'sequential_1_lstm_3_while_placeholder_2+
'sequential_1_lstm_3_while_placeholder_3F
Bsequential_1_lstm_3_while_less_sequential_1_lstm_3_strided_slice_1[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_33731___redundant_placeholder0[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_33731___redundant_placeholder1[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_33731___redundant_placeholder2[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_33731___redundant_placeholder3&
"sequential_1_lstm_3_while_identity
²
sequential_1/lstm_3/while/LessLess%sequential_1_lstm_3_while_placeholderBsequential_1_lstm_3_while_less_sequential_1_lstm_3_strided_slice_1*
T0*
_output_shapes
: s
"sequential_1/lstm_3/while/IdentityIdentity"sequential_1/lstm_3/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_1_lstm_3_while_identity+sequential_1/lstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
±¬
ß
 __inference__wrapped_model_34103
lstm_3_inputP
>sequential_1_lstm_3_lstm_cell_6_matmul_readvariableop_resource:,R
@sequential_1_lstm_3_lstm_cell_6_matmul_1_readvariableop_resource:,M
?sequential_1_lstm_3_lstm_cell_6_biasadd_readvariableop_resource:,P
>sequential_1_lstm_4_lstm_cell_7_matmul_readvariableop_resource:R
@sequential_1_lstm_4_lstm_cell_7_matmul_1_readvariableop_resource:M
?sequential_1_lstm_4_lstm_cell_7_biasadd_readvariableop_resource:P
>sequential_1_lstm_5_lstm_cell_8_matmul_readvariableop_resource:R
@sequential_1_lstm_5_lstm_cell_8_matmul_1_readvariableop_resource:M
?sequential_1_lstm_5_lstm_cell_8_biasadd_readvariableop_resource:E
3sequential_1_dense_1_matmul_readvariableop_resource:B
4sequential_1_dense_1_biasadd_readvariableop_resource:
identity¢+sequential_1/dense_1/BiasAdd/ReadVariableOp¢*sequential_1/dense_1/MatMul/ReadVariableOp¢6sequential_1/lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp¢5sequential_1/lstm_3/lstm_cell_6/MatMul/ReadVariableOp¢7sequential_1/lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp¢sequential_1/lstm_3/while¢6sequential_1/lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp¢5sequential_1/lstm_4/lstm_cell_7/MatMul/ReadVariableOp¢7sequential_1/lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp¢sequential_1/lstm_4/while¢6sequential_1/lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp¢5sequential_1/lstm_5/lstm_cell_8/MatMul/ReadVariableOp¢7sequential_1/lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp¢sequential_1/lstm_5/whileU
sequential_1/lstm_3/ShapeShapelstm_3_input*
T0*
_output_shapes
:q
'sequential_1/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_1/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_1/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!sequential_1/lstm_3/strided_sliceStridedSlice"sequential_1/lstm_3/Shape:output:00sequential_1/lstm_3/strided_slice/stack:output:02sequential_1/lstm_3/strided_slice/stack_1:output:02sequential_1/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_1/lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¯
 sequential_1/lstm_3/zeros/packedPack*sequential_1/lstm_3/strided_slice:output:0+sequential_1/lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_1/lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
sequential_1/lstm_3/zerosFill)sequential_1/lstm_3/zeros/packed:output:0(sequential_1/lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$sequential_1/lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :³
"sequential_1/lstm_3/zeros_1/packedPack*sequential_1/lstm_3/strided_slice:output:0-sequential_1/lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_1/lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
sequential_1/lstm_3/zeros_1Fill+sequential_1/lstm_3/zeros_1/packed:output:0*sequential_1/lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"sequential_1/lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
sequential_1/lstm_3/transpose	Transposelstm_3_input+sequential_1/lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
sequential_1/lstm_3/Shape_1Shape!sequential_1/lstm_3/transpose:y:0*
T0*
_output_shapes
:s
)sequential_1/lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#sequential_1/lstm_3/strided_slice_1StridedSlice$sequential_1/lstm_3/Shape_1:output:02sequential_1/lstm_3/strided_slice_1/stack:output:04sequential_1/lstm_3/strided_slice_1/stack_1:output:04sequential_1/lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_1/lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿð
!sequential_1/lstm_3/TensorArrayV2TensorListReserve8sequential_1/lstm_3/TensorArrayV2/element_shape:output:0,sequential_1/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Isequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
;sequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_3/transpose:y:0Rsequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)sequential_1/lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
#sequential_1/lstm_3/strided_slice_2StridedSlice!sequential_1/lstm_3/transpose:y:02sequential_1/lstm_3/strided_slice_2/stack:output:04sequential_1/lstm_3/strided_slice_2/stack_1:output:04sequential_1/lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask´
5sequential_1/lstm_3/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp>sequential_1_lstm_3_lstm_cell_6_matmul_readvariableop_resource*
_output_shapes

:,*
dtype0Ï
&sequential_1/lstm_3/lstm_cell_6/MatMulMatMul,sequential_1/lstm_3/strided_slice_2:output:0=sequential_1/lstm_3/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,¸
7sequential_1/lstm_3/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp@sequential_1_lstm_3_lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0É
(sequential_1/lstm_3/lstm_cell_6/MatMul_1MatMul"sequential_1/lstm_3/zeros:output:0?sequential_1/lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,Ä
#sequential_1/lstm_3/lstm_cell_6/addAddV20sequential_1/lstm_3/lstm_cell_6/MatMul:product:02sequential_1/lstm_3/lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,²
6sequential_1/lstm_3/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_lstm_3_lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0Í
'sequential_1/lstm_3/lstm_cell_6/BiasAddBiasAdd'sequential_1/lstm_3/lstm_cell_6/add:z:0>sequential_1/lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,q
/sequential_1/lstm_3/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%sequential_1/lstm_3/lstm_cell_6/splitSplit8sequential_1/lstm_3/lstm_cell_6/split/split_dim:output:00sequential_1/lstm_3/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
'sequential_1/lstm_3/lstm_cell_6/SigmoidSigmoid.sequential_1/lstm_3/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_1/lstm_3/lstm_cell_6/Sigmoid_1Sigmoid.sequential_1/lstm_3/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
#sequential_1/lstm_3/lstm_cell_6/mulMul-sequential_1/lstm_3/lstm_cell_6/Sigmoid_1:y:0$sequential_1/lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sequential_1/lstm_3/lstm_cell_6/ReluRelu.sequential_1/lstm_3/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
%sequential_1/lstm_3/lstm_cell_6/mul_1Mul+sequential_1/lstm_3/lstm_cell_6/Sigmoid:y:02sequential_1/lstm_3/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
%sequential_1/lstm_3/lstm_cell_6/add_1AddV2'sequential_1/lstm_3/lstm_cell_6/mul:z:0)sequential_1/lstm_3/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_1/lstm_3/lstm_cell_6/Sigmoid_2Sigmoid.sequential_1/lstm_3/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential_1/lstm_3/lstm_cell_6/Relu_1Relu)sequential_1/lstm_3/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
%sequential_1/lstm_3/lstm_cell_6/mul_2Mul-sequential_1/lstm_3/lstm_cell_6/Sigmoid_2:y:04sequential_1/lstm_3/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1sequential_1/lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ô
#sequential_1/lstm_3/TensorArrayV2_1TensorListReserve:sequential_1/lstm_3/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
sequential_1/lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_1/lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿh
&sequential_1/lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
sequential_1/lstm_3/whileWhile/sequential_1/lstm_3/while/loop_counter:output:05sequential_1/lstm_3/while/maximum_iterations:output:0!sequential_1/lstm_3/time:output:0,sequential_1/lstm_3/TensorArrayV2_1:handle:0"sequential_1/lstm_3/zeros:output:0$sequential_1/lstm_3/zeros_1:output:0,sequential_1/lstm_3/strided_slice_1:output:0Ksequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_1_lstm_3_lstm_cell_6_matmul_readvariableop_resource@sequential_1_lstm_3_lstm_cell_6_matmul_1_readvariableop_resource?sequential_1_lstm_3_lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$sequential_1_lstm_3_while_body_33732*0
cond(R&
$sequential_1_lstm_3_while_cond_33731*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Dsequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   þ
6sequential_1/lstm_3/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_3/while:output:3Msequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0|
)sequential_1/lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿu
+sequential_1/lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ë
#sequential_1/lstm_3/strided_slice_3StridedSlice?sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_3/strided_slice_3/stack:output:04sequential_1/lstm_3/strided_slice_3/stack_1:output:04sequential_1/lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masky
$sequential_1/lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ò
sequential_1/lstm_3/transpose_1	Transpose?sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
sequential_1/lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
sequential_1/dropout_3/IdentityIdentity#sequential_1/lstm_3/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
sequential_1/lstm_4/ShapeShape(sequential_1/dropout_3/Identity:output:0*
T0*
_output_shapes
:q
'sequential_1/lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_1/lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_1/lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!sequential_1/lstm_4/strided_sliceStridedSlice"sequential_1/lstm_4/Shape:output:00sequential_1/lstm_4/strided_slice/stack:output:02sequential_1/lstm_4/strided_slice/stack_1:output:02sequential_1/lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_1/lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¯
 sequential_1/lstm_4/zeros/packedPack*sequential_1/lstm_4/strided_slice:output:0+sequential_1/lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_1/lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
sequential_1/lstm_4/zerosFill)sequential_1/lstm_4/zeros/packed:output:0(sequential_1/lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$sequential_1/lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :³
"sequential_1/lstm_4/zeros_1/packedPack*sequential_1/lstm_4/strided_slice:output:0-sequential_1/lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_1/lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
sequential_1/lstm_4/zeros_1Fill+sequential_1/lstm_4/zeros_1/packed:output:0*sequential_1/lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"sequential_1/lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ·
sequential_1/lstm_4/transpose	Transpose(sequential_1/dropout_3/Identity:output:0+sequential_1/lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
sequential_1/lstm_4/Shape_1Shape!sequential_1/lstm_4/transpose:y:0*
T0*
_output_shapes
:s
)sequential_1/lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#sequential_1/lstm_4/strided_slice_1StridedSlice$sequential_1/lstm_4/Shape_1:output:02sequential_1/lstm_4/strided_slice_1/stack:output:04sequential_1/lstm_4/strided_slice_1/stack_1:output:04sequential_1/lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_1/lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿð
!sequential_1/lstm_4/TensorArrayV2TensorListReserve8sequential_1/lstm_4/TensorArrayV2/element_shape:output:0,sequential_1/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Isequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
;sequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_4/transpose:y:0Rsequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)sequential_1/lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
#sequential_1/lstm_4/strided_slice_2StridedSlice!sequential_1/lstm_4/transpose:y:02sequential_1/lstm_4/strided_slice_2/stack:output:04sequential_1/lstm_4/strided_slice_2/stack_1:output:04sequential_1/lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask´
5sequential_1/lstm_4/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp>sequential_1_lstm_4_lstm_cell_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ï
&sequential_1/lstm_4/lstm_cell_7/MatMulMatMul,sequential_1/lstm_4/strided_slice_2:output:0=sequential_1/lstm_4/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
7sequential_1/lstm_4/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp@sequential_1_lstm_4_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0É
(sequential_1/lstm_4/lstm_cell_7/MatMul_1MatMul"sequential_1/lstm_4/zeros:output:0?sequential_1/lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
#sequential_1/lstm_4/lstm_cell_7/addAddV20sequential_1/lstm_4/lstm_cell_7/MatMul:product:02sequential_1/lstm_4/lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
6sequential_1/lstm_4/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_lstm_4_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Í
'sequential_1/lstm_4/lstm_cell_7/BiasAddBiasAdd'sequential_1/lstm_4/lstm_cell_7/add:z:0>sequential_1/lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
/sequential_1/lstm_4/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%sequential_1/lstm_4/lstm_cell_7/splitSplit8sequential_1/lstm_4/lstm_cell_7/split/split_dim:output:00sequential_1/lstm_4/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
'sequential_1/lstm_4/lstm_cell_7/SigmoidSigmoid.sequential_1/lstm_4/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_1/lstm_4/lstm_cell_7/Sigmoid_1Sigmoid.sequential_1/lstm_4/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
#sequential_1/lstm_4/lstm_cell_7/mulMul-sequential_1/lstm_4/lstm_cell_7/Sigmoid_1:y:0$sequential_1/lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sequential_1/lstm_4/lstm_cell_7/ReluRelu.sequential_1/lstm_4/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
%sequential_1/lstm_4/lstm_cell_7/mul_1Mul+sequential_1/lstm_4/lstm_cell_7/Sigmoid:y:02sequential_1/lstm_4/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
%sequential_1/lstm_4/lstm_cell_7/add_1AddV2'sequential_1/lstm_4/lstm_cell_7/mul:z:0)sequential_1/lstm_4/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_1/lstm_4/lstm_cell_7/Sigmoid_2Sigmoid.sequential_1/lstm_4/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential_1/lstm_4/lstm_cell_7/Relu_1Relu)sequential_1/lstm_4/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
%sequential_1/lstm_4/lstm_cell_7/mul_2Mul-sequential_1/lstm_4/lstm_cell_7/Sigmoid_2:y:04sequential_1/lstm_4/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1sequential_1/lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ô
#sequential_1/lstm_4/TensorArrayV2_1TensorListReserve:sequential_1/lstm_4/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
sequential_1/lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_1/lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿh
&sequential_1/lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
sequential_1/lstm_4/whileWhile/sequential_1/lstm_4/while/loop_counter:output:05sequential_1/lstm_4/while/maximum_iterations:output:0!sequential_1/lstm_4/time:output:0,sequential_1/lstm_4/TensorArrayV2_1:handle:0"sequential_1/lstm_4/zeros:output:0$sequential_1/lstm_4/zeros_1:output:0,sequential_1/lstm_4/strided_slice_1:output:0Ksequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_1_lstm_4_lstm_cell_7_matmul_readvariableop_resource@sequential_1_lstm_4_lstm_cell_7_matmul_1_readvariableop_resource?sequential_1_lstm_4_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$sequential_1_lstm_4_while_body_33872*0
cond(R&
$sequential_1_lstm_4_while_cond_33871*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Dsequential_1/lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   þ
6sequential_1/lstm_4/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_4/while:output:3Msequential_1/lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0|
)sequential_1/lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿu
+sequential_1/lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ë
#sequential_1/lstm_4/strided_slice_3StridedSlice?sequential_1/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_4/strided_slice_3/stack:output:04sequential_1/lstm_4/strided_slice_3/stack_1:output:04sequential_1/lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masky
$sequential_1/lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ò
sequential_1/lstm_4/transpose_1	Transpose?sequential_1/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
sequential_1/lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
sequential_1/dropout_4/IdentityIdentity#sequential_1/lstm_4/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
sequential_1/lstm_5/ShapeShape(sequential_1/dropout_4/Identity:output:0*
T0*
_output_shapes
:q
'sequential_1/lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_1/lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_1/lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!sequential_1/lstm_5/strided_sliceStridedSlice"sequential_1/lstm_5/Shape:output:00sequential_1/lstm_5/strided_slice/stack:output:02sequential_1/lstm_5/strided_slice/stack_1:output:02sequential_1/lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_1/lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¯
 sequential_1/lstm_5/zeros/packedPack*sequential_1/lstm_5/strided_slice:output:0+sequential_1/lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_1/lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
sequential_1/lstm_5/zerosFill)sequential_1/lstm_5/zeros/packed:output:0(sequential_1/lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$sequential_1/lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :³
"sequential_1/lstm_5/zeros_1/packedPack*sequential_1/lstm_5/strided_slice:output:0-sequential_1/lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_1/lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
sequential_1/lstm_5/zeros_1Fill+sequential_1/lstm_5/zeros_1/packed:output:0*sequential_1/lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"sequential_1/lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ·
sequential_1/lstm_5/transpose	Transpose(sequential_1/dropout_4/Identity:output:0+sequential_1/lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
sequential_1/lstm_5/Shape_1Shape!sequential_1/lstm_5/transpose:y:0*
T0*
_output_shapes
:s
)sequential_1/lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#sequential_1/lstm_5/strided_slice_1StridedSlice$sequential_1/lstm_5/Shape_1:output:02sequential_1/lstm_5/strided_slice_1/stack:output:04sequential_1/lstm_5/strided_slice_1/stack_1:output:04sequential_1/lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_1/lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿð
!sequential_1/lstm_5/TensorArrayV2TensorListReserve8sequential_1/lstm_5/TensorArrayV2/element_shape:output:0,sequential_1/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Isequential_1/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
;sequential_1/lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_5/transpose:y:0Rsequential_1/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)sequential_1/lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
#sequential_1/lstm_5/strided_slice_2StridedSlice!sequential_1/lstm_5/transpose:y:02sequential_1/lstm_5/strided_slice_2/stack:output:04sequential_1/lstm_5/strided_slice_2/stack_1:output:04sequential_1/lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask´
5sequential_1/lstm_5/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp>sequential_1_lstm_5_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ï
&sequential_1/lstm_5/lstm_cell_8/MatMulMatMul,sequential_1/lstm_5/strided_slice_2:output:0=sequential_1/lstm_5/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
7sequential_1/lstm_5/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp@sequential_1_lstm_5_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0É
(sequential_1/lstm_5/lstm_cell_8/MatMul_1MatMul"sequential_1/lstm_5/zeros:output:0?sequential_1/lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
#sequential_1/lstm_5/lstm_cell_8/addAddV20sequential_1/lstm_5/lstm_cell_8/MatMul:product:02sequential_1/lstm_5/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
6sequential_1/lstm_5/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_lstm_5_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Í
'sequential_1/lstm_5/lstm_cell_8/BiasAddBiasAdd'sequential_1/lstm_5/lstm_cell_8/add:z:0>sequential_1/lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
/sequential_1/lstm_5/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%sequential_1/lstm_5/lstm_cell_8/splitSplit8sequential_1/lstm_5/lstm_cell_8/split/split_dim:output:00sequential_1/lstm_5/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
'sequential_1/lstm_5/lstm_cell_8/SigmoidSigmoid.sequential_1/lstm_5/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_1/lstm_5/lstm_cell_8/Sigmoid_1Sigmoid.sequential_1/lstm_5/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
#sequential_1/lstm_5/lstm_cell_8/mulMul-sequential_1/lstm_5/lstm_cell_8/Sigmoid_1:y:0$sequential_1/lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sequential_1/lstm_5/lstm_cell_8/ReluRelu.sequential_1/lstm_5/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
%sequential_1/lstm_5/lstm_cell_8/mul_1Mul+sequential_1/lstm_5/lstm_cell_8/Sigmoid:y:02sequential_1/lstm_5/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
%sequential_1/lstm_5/lstm_cell_8/add_1AddV2'sequential_1/lstm_5/lstm_cell_8/mul:z:0)sequential_1/lstm_5/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_1/lstm_5/lstm_cell_8/Sigmoid_2Sigmoid.sequential_1/lstm_5/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential_1/lstm_5/lstm_cell_8/Relu_1Relu)sequential_1/lstm_5/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
%sequential_1/lstm_5/lstm_cell_8/mul_2Mul-sequential_1/lstm_5/lstm_cell_8/Sigmoid_2:y:04sequential_1/lstm_5/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1sequential_1/lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ô
#sequential_1/lstm_5/TensorArrayV2_1TensorListReserve:sequential_1/lstm_5/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
sequential_1/lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_1/lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿh
&sequential_1/lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
sequential_1/lstm_5/whileWhile/sequential_1/lstm_5/while/loop_counter:output:05sequential_1/lstm_5/while/maximum_iterations:output:0!sequential_1/lstm_5/time:output:0,sequential_1/lstm_5/TensorArrayV2_1:handle:0"sequential_1/lstm_5/zeros:output:0$sequential_1/lstm_5/zeros_1:output:0,sequential_1/lstm_5/strided_slice_1:output:0Ksequential_1/lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_1_lstm_5_lstm_cell_8_matmul_readvariableop_resource@sequential_1_lstm_5_lstm_cell_8_matmul_1_readvariableop_resource?sequential_1_lstm_5_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$sequential_1_lstm_5_while_body_34012*0
cond(R&
$sequential_1_lstm_5_while_cond_34011*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Dsequential_1/lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   þ
6sequential_1/lstm_5/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_5/while:output:3Msequential_1/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0|
)sequential_1/lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿu
+sequential_1/lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ë
#sequential_1/lstm_5/strided_slice_3StridedSlice?sequential_1/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_5/strided_slice_3/stack:output:04sequential_1/lstm_5/strided_slice_3/stack_1:output:04sequential_1/lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masky
$sequential_1/lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ò
sequential_1/lstm_5/transpose_1	Transpose?sequential_1/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
sequential_1/lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
sequential_1/dropout_5/IdentityIdentity,sequential_1/lstm_5/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0µ
sequential_1/dense_1/MatMulMatMul(sequential_1/dropout_5/Identity:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
IdentityIdentity%sequential_1/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
NoOpNoOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp7^sequential_1/lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp6^sequential_1/lstm_3/lstm_cell_6/MatMul/ReadVariableOp8^sequential_1/lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp^sequential_1/lstm_3/while7^sequential_1/lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp6^sequential_1/lstm_4/lstm_cell_7/MatMul/ReadVariableOp8^sequential_1/lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp^sequential_1/lstm_4/while7^sequential_1/lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp6^sequential_1/lstm_5/lstm_cell_8/MatMul/ReadVariableOp8^sequential_1/lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp^sequential_1/lstm_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2p
6sequential_1/lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp6sequential_1/lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp2n
5sequential_1/lstm_3/lstm_cell_6/MatMul/ReadVariableOp5sequential_1/lstm_3/lstm_cell_6/MatMul/ReadVariableOp2r
7sequential_1/lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp7sequential_1/lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp26
sequential_1/lstm_3/whilesequential_1/lstm_3/while2p
6sequential_1/lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp6sequential_1/lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp2n
5sequential_1/lstm_4/lstm_cell_7/MatMul/ReadVariableOp5sequential_1/lstm_4/lstm_cell_7/MatMul/ReadVariableOp2r
7sequential_1/lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp7sequential_1/lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp26
sequential_1/lstm_4/whilesequential_1/lstm_4/while2p
6sequential_1/lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp6sequential_1/lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp2n
5sequential_1/lstm_5/lstm_cell_8/MatMul/ReadVariableOp5sequential_1/lstm_5/lstm_cell_8/MatMul/ReadVariableOp2r
7sequential_1/lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp7sequential_1/lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp26
sequential_1/lstm_5/whilesequential_1/lstm_5/while:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_3_input
Í7
À
while_body_35778
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_8_matmul_readvariableop_resource_0:F
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:A
3while_lstm_cell_8_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_8_matmul_readvariableop_resource:D
2while_lstm_cell_8_matmul_1_readvariableop_resource:?
1while_lstm_cell_8_biasadd_readvariableop_resource:¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0·
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0£
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
á
Î
$sequential_1_lstm_5_while_cond_34011D
@sequential_1_lstm_5_while_sequential_1_lstm_5_while_loop_counterJ
Fsequential_1_lstm_5_while_sequential_1_lstm_5_while_maximum_iterations)
%sequential_1_lstm_5_while_placeholder+
'sequential_1_lstm_5_while_placeholder_1+
'sequential_1_lstm_5_while_placeholder_2+
'sequential_1_lstm_5_while_placeholder_3F
Bsequential_1_lstm_5_while_less_sequential_1_lstm_5_strided_slice_1[
Wsequential_1_lstm_5_while_sequential_1_lstm_5_while_cond_34011___redundant_placeholder0[
Wsequential_1_lstm_5_while_sequential_1_lstm_5_while_cond_34011___redundant_placeholder1[
Wsequential_1_lstm_5_while_sequential_1_lstm_5_while_cond_34011___redundant_placeholder2[
Wsequential_1_lstm_5_while_sequential_1_lstm_5_while_cond_34011___redundant_placeholder3&
"sequential_1_lstm_5_while_identity
²
sequential_1/lstm_5/while/LessLess%sequential_1_lstm_5_while_placeholderBsequential_1_lstm_5_while_less_sequential_1_lstm_5_strided_slice_1*
T0*
_output_shapes
: s
"sequential_1/lstm_5/while/IdentityIdentity"sequential_1/lstm_5/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_1_lstm_5_while_identity+sequential_1/lstm_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Í7
À
while_body_38429
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_7_matmul_readvariableop_resource_0:F
4while_lstm_cell_7_matmul_1_readvariableop_resource_0:A
3while_lstm_cell_7_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_7_matmul_readvariableop_resource:D
2while_lstm_cell_7_matmul_1_readvariableop_resource:?
1while_lstm_cell_7_biasadd_readvariableop_resource:¢(while/lstm_cell_7/BiasAdd/ReadVariableOp¢'while/lstm_cell_7/MatMul/ReadVariableOp¢)while/lstm_cell_7/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0·
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0£
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_7/BiasAdd/ReadVariableOp(^while/lstm_cell_7/MatMul/ReadVariableOp*^while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_7/BiasAdd/ReadVariableOp(while/lstm_cell_7/BiasAdd/ReadVariableOp2R
'while/lstm_cell_7/MatMul/ReadVariableOp'while/lstm_cell_7/MatMul/ReadVariableOp2V
)while/lstm_cell_7/MatMul_1/ReadVariableOp)while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Í7
À
while_body_38929
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_8_matmul_readvariableop_resource_0:F
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:A
3while_lstm_cell_8_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_8_matmul_readvariableop_resource:D
2while_lstm_cell_8_matmul_1_readvariableop_resource:?
1while_lstm_cell_8_biasadd_readvariableop_resource:¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0·
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0£
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ßI

A__inference_lstm_5_layer_call_and_return_conditional_losses_35616

inputs<
*lstm_cell_8_matmul_readvariableop_resource:>
,lstm_cell_8_matmul_1_readvariableop_resource:9
+lstm_cell_8_biasadd_readvariableop_resource:
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35532*
condR
while_cond_35531*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í7
À
while_body_37929
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_6_matmul_readvariableop_resource_0:,F
4while_lstm_cell_6_matmul_1_readvariableop_resource_0:,A
3while_lstm_cell_6_biasadd_readvariableop_resource_0:,
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_6_matmul_readvariableop_resource:,D
2while_lstm_cell_6_matmul_1_readvariableop_resource:,?
1while_lstm_cell_6_biasadd_readvariableop_resource:,¢(while/lstm_cell_6/BiasAdd/ReadVariableOp¢'while/lstm_cell_6/MatMul/ReadVariableOp¢)while/lstm_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:,*
dtype0·
while/lstm_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
)while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:,*
dtype0
while/lstm_cell_6/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
while/lstm_cell_6/addAddV2"while/lstm_cell_6/MatMul:product:0$while/lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
(while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:,*
dtype0£
while/lstm_cell_6/BiasAddBiasAddwhile/lstm_cell_6/add:z:00while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,c
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0"while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_6/SigmoidSigmoid while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_6/Sigmoid_1Sigmoid while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mulMulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_6/ReluRelu while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mul_1Mulwhile/lstm_cell_6/Sigmoid:y:0$while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/add_1AddV2while/lstm_cell_6/mul:z:0while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_6/Sigmoid_2Sigmoid while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_6/Relu_1Reluwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mul_2Mulwhile/lstm_cell_6/Sigmoid_2:y:0&while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_6/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_6/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_6/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_6_biasadd_readvariableop_resource3while_lstm_cell_6_biasadd_readvariableop_resource_0"j
2while_lstm_cell_6_matmul_1_readvariableop_resource4while_lstm_cell_6_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_6_matmul_readvariableop_resource2while_lstm_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_6/BiasAdd/ReadVariableOp(while/lstm_cell_6/BiasAdd/ReadVariableOp2R
'while/lstm_cell_6/MatMul/ReadVariableOp'while/lstm_cell_6/MatMul/ReadVariableOp2V
)while/lstm_cell_6/MatMul_1/ReadVariableOp)while/lstm_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°
¾
while_cond_35217
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_35217___redundant_placeholder03
/while_while_cond_35217___redundant_placeholder13
/while_while_cond_35217___redundant_placeholder23
/while_while_cond_35217___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ø
°
&__inference_lstm_4_layer_call_fn_38073

inputs
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_35459s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
¾
while_cond_38928
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_38928___redundant_placeholder03
/while_while_cond_38928___redundant_placeholder13
/while_while_cond_38928___redundant_placeholder23
/while_while_cond_38928___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ï?
 

lstm_4_while_body_37123*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0K
9lstm_4_while_lstm_cell_7_matmul_readvariableop_resource_0:M
;lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource_0:H
:lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource_0:
lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensorI
7lstm_4_while_lstm_cell_7_matmul_readvariableop_resource:K
9lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource:F
8lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource:¢/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp¢.lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp¢0lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   É
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.lstm_4/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp9lstm_4_while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ì
lstm_4/while/lstm_cell_7/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
0lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp;lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0³
!lstm_4/while/lstm_cell_7/MatMul_1MatMullstm_4_while_placeholder_28lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
lstm_4/while/lstm_cell_7/addAddV2)lstm_4/while/lstm_cell_7/MatMul:product:0+lstm_4/while/lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0¸
 lstm_4/while/lstm_cell_7/BiasAddBiasAdd lstm_4/while/lstm_cell_7/add:z:07lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(lstm_4/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_4/while/lstm_cell_7/splitSplit1lstm_4/while/lstm_cell_7/split/split_dim:output:0)lstm_4/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 lstm_4/while/lstm_cell_7/SigmoidSigmoid'lstm_4/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_4/while/lstm_cell_7/Sigmoid_1Sigmoid'lstm_4/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/while/lstm_cell_7/mulMul&lstm_4/while/lstm_cell_7/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/while/lstm_cell_7/ReluRelu'lstm_4/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
lstm_4/while/lstm_cell_7/mul_1Mul$lstm_4/while/lstm_cell_7/Sigmoid:y:0+lstm_4/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/while/lstm_cell_7/add_1AddV2 lstm_4/while/lstm_cell_7/mul:z:0"lstm_4/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_4/while/lstm_cell_7/Sigmoid_2Sigmoid'lstm_4/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_4/while/lstm_cell_7/Relu_1Relu"lstm_4/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
lstm_4/while/lstm_cell_7/mul_2Mul&lstm_4/while/lstm_cell_7/Sigmoid_2:y:0-lstm_4/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
1lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_4_while_placeholder_1lstm_4_while_placeholder"lstm_4/while/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_4/while/addAddV2lstm_4_while_placeholderlstm_4/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_4/while/add_1AddV2&lstm_4_while_lstm_4_while_loop_counterlstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_4/while/IdentityIdentitylstm_4/while/add_1:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 
lstm_4/while/Identity_1Identity,lstm_4_while_lstm_4_while_maximum_iterations^lstm_4/while/NoOp*
T0*
_output_shapes
: n
lstm_4/while/Identity_2Identitylstm_4/while/add:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: ®
lstm_4/while/Identity_3IdentityAlstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_4/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_4/while/Identity_4Identity"lstm_4/while/lstm_cell_7/mul_2:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_7/add_1:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
lstm_4/while/NoOpNoOp0^lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp/^lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp1^lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_4_while_identitylstm_4/while/Identity:output:0";
lstm_4_while_identity_1 lstm_4/while/Identity_1:output:0";
lstm_4_while_identity_2 lstm_4/while/Identity_2:output:0";
lstm_4_while_identity_3 lstm_4/while/Identity_3:output:0";
lstm_4_while_identity_4 lstm_4/while/Identity_4:output:0";
lstm_4_while_identity_5 lstm_4/while/Identity_5:output:0"L
#lstm_4_while_lstm_4_strided_slice_1%lstm_4_while_lstm_4_strided_slice_1_0"v
8lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource:lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource_0"x
9lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource;lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource_0"t
7lstm_4_while_lstm_cell_7_matmul_readvariableop_resource9lstm_4_while_lstm_cell_7_matmul_readvariableop_resource_0"Ä
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensoralstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2b
/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp2`
.lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp.lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp2d
0lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp0lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ç
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_38671

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î7
ù
A__inference_lstm_5_layer_call_and_return_conditional_losses_35144

inputs#
lstm_cell_8_35062:#
lstm_cell_8_35064:
lstm_cell_8_35066:
identity¢#lstm_cell_8/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskì
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_35062lstm_cell_8_35064lstm_cell_8_35066*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_35016n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¯
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_35062lstm_cell_8_35064lstm_cell_8_35066*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35075*
condR
while_cond_35074*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
NoOpNoOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
¾
while_cond_35374
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_35374___redundant_placeholder03
/while_while_cond_35374___redundant_placeholder13
/while_while_cond_35374___redundant_placeholder23
/while_while_cond_35374___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ä
ñ
+__inference_lstm_cell_6_layer_call_fn_39362

inputs
states_0
states_1
unknown:,
	unknown_0:,
	unknown_1:,
identity

identity_1

identity_2¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_34170o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
úù
»

G__inference_sequential_1_layer_call_and_return_conditional_losses_36917

inputsC
1lstm_3_lstm_cell_6_matmul_readvariableop_resource:,E
3lstm_3_lstm_cell_6_matmul_1_readvariableop_resource:,@
2lstm_3_lstm_cell_6_biasadd_readvariableop_resource:,C
1lstm_4_lstm_cell_7_matmul_readvariableop_resource:E
3lstm_4_lstm_cell_7_matmul_1_readvariableop_resource:@
2lstm_4_lstm_cell_7_biasadd_readvariableop_resource:C
1lstm_5_lstm_cell_8_matmul_readvariableop_resource:E
3lstm_5_lstm_cell_8_matmul_1_readvariableop_resource:@
2lstm_5_lstm_cell_8_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢)lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp¢(lstm_3/lstm_cell_6/MatMul/ReadVariableOp¢*lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp¢lstm_3/while¢)lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp¢(lstm_4/lstm_cell_7/MatMul/ReadVariableOp¢*lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp¢lstm_4/while¢)lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp¢(lstm_5/lstm_cell_8/MatMul/ReadVariableOp¢*lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp¢lstm_5/whileB
lstm_3/ShapeShapeinputs*
T0*
_output_shapes
:d
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          {
lstm_3/transpose	Transposeinputslstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:f
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   õ
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(lstm_3/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp1lstm_3_lstm_cell_6_matmul_readvariableop_resource*
_output_shapes

:,*
dtype0¨
lstm_3/lstm_cell_6/MatMulMatMullstm_3/strided_slice_2:output:00lstm_3/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
*lstm_3/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp3lstm_3_lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0¢
lstm_3/lstm_cell_6/MatMul_1MatMullstm_3/zeros:output:02lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
lstm_3/lstm_cell_6/addAddV2#lstm_3/lstm_cell_6/MatMul:product:0%lstm_3/lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
)lstm_3/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0¦
lstm_3/lstm_cell_6/BiasAddBiasAddlstm_3/lstm_cell_6/add:z:01lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,d
"lstm_3/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
lstm_3/lstm_cell_6/splitSplit+lstm_3/lstm_cell_6/split/split_dim:output:0#lstm_3/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
lstm_3/lstm_cell_6/SigmoidSigmoid!lstm_3/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_3/lstm_cell_6/Sigmoid_1Sigmoid!lstm_3/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/lstm_cell_6/mulMul lstm_3/lstm_cell_6/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm_3/lstm_cell_6/ReluRelu!lstm_3/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/lstm_cell_6/mul_1Mullstm_3/lstm_cell_6/Sigmoid:y:0%lstm_3/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/lstm_cell_6/add_1AddV2lstm_3/lstm_cell_6/mul:z:0lstm_3/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_3/lstm_cell_6/Sigmoid_2Sigmoid!lstm_3/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_3/lstm_cell_6/Relu_1Relulstm_3/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_3/lstm_cell_6/mul_2Mul lstm_3/lstm_cell_6/Sigmoid_2:y:0'lstm_3/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Í
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ß
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_3_lstm_cell_6_matmul_readvariableop_resource3lstm_3_lstm_cell_6_matmul_1_readvariableop_resource2lstm_3_lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_3_while_body_36546*#
condR
lstm_3_while_cond_36545*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0o
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskl
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
dropout_3/IdentityIdentitylstm_3/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
lstm_4/ShapeShapedropout_3/Identity:output:0*
T0*
_output_shapes
:d
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_4/transpose	Transposedropout_3/Identity:output:0lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
lstm_4/Shape_1Shapelstm_4/transpose:y:0*
T0*
_output_shapes
:f
lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_4/strided_slice_1StridedSlicelstm_4/Shape_1:output:0%lstm_4/strided_slice_1/stack:output:0'lstm_4/strided_slice_1/stack_1:output:0'lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_4/TensorArrayV2TensorListReserve+lstm_4/TensorArrayV2/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   õ
.lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_4/transpose:y:0Elstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_4/strided_slice_2StridedSlicelstm_4/transpose:y:0%lstm_4/strided_slice_2/stack:output:0'lstm_4/strided_slice_2/stack_1:output:0'lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(lstm_4/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp1lstm_4_lstm_cell_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¨
lstm_4/lstm_cell_7/MatMulMatMullstm_4/strided_slice_2:output:00lstm_4/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_4/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp3lstm_4_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0¢
lstm_4/lstm_cell_7/MatMul_1MatMullstm_4/zeros:output:02lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/lstm_cell_7/addAddV2#lstm_4/lstm_cell_7/MatMul:product:0%lstm_4/lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)lstm_4/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
lstm_4/lstm_cell_7/BiasAddBiasAddlstm_4/lstm_cell_7/add:z:01lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"lstm_4/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
lstm_4/lstm_cell_7/splitSplit+lstm_4/lstm_cell_7/split/split_dim:output:0#lstm_4/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
lstm_4/lstm_cell_7/SigmoidSigmoid!lstm_4/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_4/lstm_cell_7/Sigmoid_1Sigmoid!lstm_4/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/lstm_cell_7/mulMul lstm_4/lstm_cell_7/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm_4/lstm_cell_7/ReluRelu!lstm_4/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/lstm_cell_7/mul_1Mullstm_4/lstm_cell_7/Sigmoid:y:0%lstm_4/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/lstm_cell_7/add_1AddV2lstm_4/lstm_cell_7/mul:z:0lstm_4/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_4/lstm_cell_7/Sigmoid_2Sigmoid!lstm_4/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_4/lstm_cell_7/Relu_1Relulstm_4/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/lstm_cell_7/mul_2Mul lstm_4/lstm_cell_7/Sigmoid_2:y:0'lstm_4/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Í
lstm_4/TensorArrayV2_1TensorListReserve-lstm_4/TensorArrayV2_1/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ß
lstm_4/whileWhile"lstm_4/while/loop_counter:output:0(lstm_4/while/maximum_iterations:output:0lstm_4/time:output:0lstm_4/TensorArrayV2_1:handle:0lstm_4/zeros:output:0lstm_4/zeros_1:output:0lstm_4/strided_slice_1:output:0>lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_4_lstm_cell_7_matmul_readvariableop_resource3lstm_4_lstm_cell_7_matmul_1_readvariableop_resource2lstm_4_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_4_while_body_36686*#
condR
lstm_4_while_cond_36685*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0o
lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_4/strided_slice_3StridedSlice2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_4/strided_slice_3/stack:output:0'lstm_4/strided_slice_3/stack_1:output:0'lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskl
lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_4/transpose_1	Transpose2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
dropout_4/IdentityIdentitylstm_4/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
lstm_5/ShapeShapedropout_4/Identity:output:0*
T0*
_output_shapes
:d
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_5/transpose	Transposedropout_4/Identity:output:0lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:f
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   õ
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(lstm_5/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¨
lstm_5/lstm_cell_8/MatMulMatMullstm_5/strided_slice_2:output:00lstm_5/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_5/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0¢
lstm_5/lstm_cell_8/MatMul_1MatMullstm_5/zeros:output:02lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/lstm_cell_8/addAddV2#lstm_5/lstm_cell_8/MatMul:product:0%lstm_5/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)lstm_5/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp2lstm_5_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
lstm_5/lstm_cell_8/BiasAddBiasAddlstm_5/lstm_cell_8/add:z:01lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"lstm_5/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
lstm_5/lstm_cell_8/splitSplit+lstm_5/lstm_cell_8/split/split_dim:output:0#lstm_5/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
lstm_5/lstm_cell_8/SigmoidSigmoid!lstm_5/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_5/lstm_cell_8/Sigmoid_1Sigmoid!lstm_5/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/lstm_cell_8/mulMul lstm_5/lstm_cell_8/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm_5/lstm_cell_8/ReluRelu!lstm_5/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/lstm_cell_8/mul_1Mullstm_5/lstm_cell_8/Sigmoid:y:0%lstm_5/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/lstm_cell_8/add_1AddV2lstm_5/lstm_cell_8/mul:z:0lstm_5/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_5/lstm_cell_8/Sigmoid_2Sigmoid!lstm_5/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_5/lstm_cell_8/Relu_1Relulstm_5/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_5/lstm_cell_8/mul_2Mul lstm_5/lstm_cell_8/Sigmoid_2:y:0'lstm_5/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Í
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ß
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_5_lstm_cell_8_matmul_readvariableop_resource3lstm_5_lstm_cell_8_matmul_1_readvariableop_resource2lstm_5_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_5_while_body_36826*#
condR
lstm_5_while_cond_36825*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0o
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskl
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    q
dropout_5/IdentityIdentitylstm_5/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_1/MatMulMatMuldropout_5/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp)^lstm_3/lstm_cell_6/MatMul/ReadVariableOp+^lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp^lstm_3/while*^lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp)^lstm_4/lstm_cell_7/MatMul/ReadVariableOp+^lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp^lstm_4/while*^lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp)^lstm_5/lstm_cell_8/MatMul/ReadVariableOp+^lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp^lstm_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp)lstm_3/lstm_cell_6/BiasAdd/ReadVariableOp2T
(lstm_3/lstm_cell_6/MatMul/ReadVariableOp(lstm_3/lstm_cell_6/MatMul/ReadVariableOp2X
*lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp*lstm_3/lstm_cell_6/MatMul_1/ReadVariableOp2
lstm_3/whilelstm_3/while2V
)lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp)lstm_4/lstm_cell_7/BiasAdd/ReadVariableOp2T
(lstm_4/lstm_cell_7/MatMul/ReadVariableOp(lstm_4/lstm_cell_7/MatMul/ReadVariableOp2X
*lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp*lstm_4/lstm_cell_7/MatMul_1/ReadVariableOp2
lstm_4/whilelstm_4/while2V
)lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp)lstm_5/lstm_cell_8/BiasAdd/ReadVariableOp2T
(lstm_5/lstm_cell_8/MatMul/ReadVariableOp(lstm_5/lstm_cell_8/MatMul/ReadVariableOp2X
*lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp*lstm_5/lstm_cell_8/MatMul_1/ReadVariableOp2
lstm_5/whilelstm_5/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô	
Ê
lstm_4_while_cond_37122*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1A
=lstm_4_while_lstm_4_while_cond_37122___redundant_placeholder0A
=lstm_4_while_lstm_4_while_cond_37122___redundant_placeholder1A
=lstm_4_while_lstm_4_while_cond_37122___redundant_placeholder2A
=lstm_4_while_lstm_4_while_cond_37122___redundant_placeholder3
lstm_4_while_identity
~
lstm_4/while/LessLesslstm_4_while_placeholder(lstm_4_while_less_lstm_4_strided_slice_1*
T0*
_output_shapes
: Y
lstm_4/while/IdentityIdentitylstm_4/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_4_while_identitylstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ï
b
)__inference_dropout_5_layer_call_fn_39309

inputs
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_35703o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
¾
while_cond_37785
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_37785___redundant_placeholder03
/while_while_cond_37785___redundant_placeholder13
/while_while_cond_37785___redundant_placeholder23
/while_while_cond_37785___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ø
°
&__inference_lstm_4_layer_call_fn_38084

inputs
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_36050s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
ñ
+__inference_lstm_cell_6_layer_call_fn_39379

inputs
states_0
states_1
unknown:,
	unknown_0:,
	unknown_1:,
identity

identity_1

identity_2¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_34316o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ð


,__inference_sequential_1_layer_call_fn_36487

inputs
unknown:,
	unknown_0:,
	unknown_1:,
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_36309o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
¾
while_cond_38571
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_38571___redundant_placeholder03
/while_while_cond_38571___redundant_placeholder13
/while_while_cond_38571___redundant_placeholder23
/while_while_cond_38571___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


c
D__inference_dropout_4_layer_call_and_return_conditional_losses_35891

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
J

A__inference_lstm_3_layer_call_and_return_conditional_losses_37584
inputs_0<
*lstm_cell_6_matmul_readvariableop_resource:,>
,lstm_cell_6_matmul_1_readvariableop_resource:,9
+lstm_cell_6_biasadd_readvariableop_resource:,
identity¢"lstm_cell_6/BiasAdd/ReadVariableOp¢!lstm_cell_6/MatMul/ReadVariableOp¢#lstm_cell_6/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_6/MatMul/ReadVariableOpReadVariableOp*lstm_cell_6_matmul_readvariableop_resource*
_output_shapes

:,*
dtype0
lstm_cell_6/MatMulMatMulstrided_slice_2:output:0)lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
#lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0
lstm_cell_6/MatMul_1MatMulzeros:output:0+lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
lstm_cell_6/addAddV2lstm_cell_6/MatMul:product:0lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
"lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0
lstm_cell_6/BiasAddBiasAddlstm_cell_6/add:z:0*lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,]
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_6/SigmoidSigmoidlstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_6/mulMullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_6/ReluRelulstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_6/mul_1Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_6/add_1AddV2lstm_cell_6/mul:z:0lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_6/Relu_1Relulstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_6/mul_2Mullstm_cell_6/Sigmoid_2:y:0 lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_6_matmul_readvariableop_resource,lstm_cell_6_matmul_1_readvariableop_resource+lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_37500*
condR
while_cond_37499*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_6/BiasAdd/ReadVariableOp"^lstm_cell_6/MatMul/ReadVariableOp$^lstm_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_6/BiasAdd/ReadVariableOp"lstm_cell_6/BiasAdd/ReadVariableOp2F
!lstm_cell_6/MatMul/ReadVariableOp!lstm_cell_6/MatMul/ReadVariableOp2J
#lstm_cell_6/MatMul_1/ReadVariableOp#lstm_cell_6/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Å	
ó
B__inference_dense_1_layer_call_and_return_conditional_losses_35641

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â


,__inference_sequential_1_layer_call_fn_36361
lstm_3_input
unknown:,
	unknown_0:,
	unknown_1:,
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCalllstm_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_36309o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_3_input
ä
ñ
+__inference_lstm_cell_8_layer_call_fn_39558

inputs
states_0
states_1
unknown:
	unknown_0:
	unknown_1:
identity

identity_1

identity_2¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_34870o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
É
ÿ
F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_34666

inputs

states
states_10
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ð


,__inference_sequential_1_layer_call_fn_36460

inputs
unknown:,
	unknown_0:,
	unknown_1:,
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_35648o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ	
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_35703

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô	
Ê
lstm_3_while_cond_36975*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1A
=lstm_3_while_lstm_3_while_cond_36975___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_36975___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_36975___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_36975___redundant_placeholder3
lstm_3_while_identity
~
lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: Y
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_3_while_identitylstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ø
°
&__inference_lstm_3_layer_call_fn_37430

inputs
unknown:,
	unknown_0:,
	unknown_1:,
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_35302s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
¾
while_cond_34533
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_34533___redundant_placeholder03
/while_while_cond_34533___redundant_placeholder13
/while_while_cond_34533___redundant_placeholder23
/while_while_cond_34533___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
­
E
)__inference_dropout_3_layer_call_fn_38018

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_35315d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í7
À
while_body_35218
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_6_matmul_readvariableop_resource_0:,F
4while_lstm_cell_6_matmul_1_readvariableop_resource_0:,A
3while_lstm_cell_6_biasadd_readvariableop_resource_0:,
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_6_matmul_readvariableop_resource:,D
2while_lstm_cell_6_matmul_1_readvariableop_resource:,?
1while_lstm_cell_6_biasadd_readvariableop_resource:,¢(while/lstm_cell_6/BiasAdd/ReadVariableOp¢'while/lstm_cell_6/MatMul/ReadVariableOp¢)while/lstm_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:,*
dtype0·
while/lstm_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
)while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:,*
dtype0
while/lstm_cell_6/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
while/lstm_cell_6/addAddV2"while/lstm_cell_6/MatMul:product:0$while/lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
(while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:,*
dtype0£
while/lstm_cell_6/BiasAddBiasAddwhile/lstm_cell_6/add:z:00while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,c
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0"while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_6/SigmoidSigmoid while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_6/Sigmoid_1Sigmoid while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mulMulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_6/ReluRelu while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mul_1Mulwhile/lstm_cell_6/Sigmoid:y:0$while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/add_1AddV2while/lstm_cell_6/mul:z:0while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_6/Sigmoid_2Sigmoid while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_6/Relu_1Reluwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mul_2Mulwhile/lstm_cell_6/Sigmoid_2:y:0&while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_6/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_6/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_6/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_6_biasadd_readvariableop_resource3while_lstm_cell_6_biasadd_readvariableop_resource_0"j
2while_lstm_cell_6_matmul_1_readvariableop_resource4while_lstm_cell_6_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_6_matmul_readvariableop_resource2while_lstm_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_6/BiasAdd/ReadVariableOp(while/lstm_cell_6/BiasAdd/ReadVariableOp2R
'while/lstm_cell_6/MatMul/ReadVariableOp'while/lstm_cell_6/MatMul/ReadVariableOp2V
)while/lstm_cell_6/MatMul_1/ReadVariableOp)while/lstm_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
O
À
$sequential_1_lstm_4_while_body_33872D
@sequential_1_lstm_4_while_sequential_1_lstm_4_while_loop_counterJ
Fsequential_1_lstm_4_while_sequential_1_lstm_4_while_maximum_iterations)
%sequential_1_lstm_4_while_placeholder+
'sequential_1_lstm_4_while_placeholder_1+
'sequential_1_lstm_4_while_placeholder_2+
'sequential_1_lstm_4_while_placeholder_3C
?sequential_1_lstm_4_while_sequential_1_lstm_4_strided_slice_1_0
{sequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensor_0X
Fsequential_1_lstm_4_while_lstm_cell_7_matmul_readvariableop_resource_0:Z
Hsequential_1_lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource_0:U
Gsequential_1_lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource_0:&
"sequential_1_lstm_4_while_identity(
$sequential_1_lstm_4_while_identity_1(
$sequential_1_lstm_4_while_identity_2(
$sequential_1_lstm_4_while_identity_3(
$sequential_1_lstm_4_while_identity_4(
$sequential_1_lstm_4_while_identity_5A
=sequential_1_lstm_4_while_sequential_1_lstm_4_strided_slice_1}
ysequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensorV
Dsequential_1_lstm_4_while_lstm_cell_7_matmul_readvariableop_resource:X
Fsequential_1_lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource:S
Esequential_1_lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource:¢<sequential_1/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp¢;sequential_1/lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp¢=sequential_1/lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp
Ksequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
=sequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_4_while_placeholderTsequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Â
;sequential_1/lstm_4/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOpFsequential_1_lstm_4_while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0ó
,sequential_1/lstm_4/while/lstm_cell_7/MatMulMatMulDsequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_1/lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
=sequential_1/lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOpHsequential_1_lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Ú
.sequential_1/lstm_4/while/lstm_cell_7/MatMul_1MatMul'sequential_1_lstm_4_while_placeholder_2Esequential_1/lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
)sequential_1/lstm_4/while/lstm_cell_7/addAddV26sequential_1/lstm_4/while/lstm_cell_7/MatMul:product:08sequential_1/lstm_4/while/lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
<sequential_1/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOpGsequential_1_lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0ß
-sequential_1/lstm_4/while/lstm_cell_7/BiasAddBiasAdd-sequential_1/lstm_4/while/lstm_cell_7/add:z:0Dsequential_1/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
5sequential_1/lstm_4/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¨
+sequential_1/lstm_4/while/lstm_cell_7/splitSplit>sequential_1/lstm_4/while/lstm_cell_7/split/split_dim:output:06sequential_1/lstm_4/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split 
-sequential_1/lstm_4/while/lstm_cell_7/SigmoidSigmoid4sequential_1/lstm_4/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
/sequential_1/lstm_4/while/lstm_cell_7/Sigmoid_1Sigmoid4sequential_1/lstm_4/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
)sequential_1/lstm_4/while/lstm_cell_7/mulMul3sequential_1/lstm_4/while/lstm_cell_7/Sigmoid_1:y:0'sequential_1_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_1/lstm_4/while/lstm_cell_7/ReluRelu4sequential_1/lstm_4/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
+sequential_1/lstm_4/while/lstm_cell_7/mul_1Mul1sequential_1/lstm_4/while/lstm_cell_7/Sigmoid:y:08sequential_1/lstm_4/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
+sequential_1/lstm_4/while/lstm_cell_7/add_1AddV2-sequential_1/lstm_4/while/lstm_cell_7/mul:z:0/sequential_1/lstm_4/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
/sequential_1/lstm_4/while/lstm_cell_7/Sigmoid_2Sigmoid4sequential_1/lstm_4/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_1/lstm_4/while/lstm_cell_7/Relu_1Relu/sequential_1/lstm_4/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
+sequential_1/lstm_4/while/lstm_cell_7/mul_2Mul3sequential_1/lstm_4/while/lstm_cell_7/Sigmoid_2:y:0:sequential_1/lstm_4/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>sequential_1/lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_4_while_placeholder_1%sequential_1_lstm_4_while_placeholder/sequential_1/lstm_4/while/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒa
sequential_1/lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_1/lstm_4/while/addAddV2%sequential_1_lstm_4_while_placeholder(sequential_1/lstm_4/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_1/lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :·
sequential_1/lstm_4/while/add_1AddV2@sequential_1_lstm_4_while_sequential_1_lstm_4_while_loop_counter*sequential_1/lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_1/lstm_4/while/IdentityIdentity#sequential_1/lstm_4/while/add_1:z:0^sequential_1/lstm_4/while/NoOp*
T0*
_output_shapes
: º
$sequential_1/lstm_4/while/Identity_1IdentityFsequential_1_lstm_4_while_sequential_1_lstm_4_while_maximum_iterations^sequential_1/lstm_4/while/NoOp*
T0*
_output_shapes
: 
$sequential_1/lstm_4/while/Identity_2Identity!sequential_1/lstm_4/while/add:z:0^sequential_1/lstm_4/while/NoOp*
T0*
_output_shapes
: Õ
$sequential_1/lstm_4/while/Identity_3IdentityNsequential_1/lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/lstm_4/while/NoOp*
T0*
_output_shapes
: :éèÒ´
$sequential_1/lstm_4/while/Identity_4Identity/sequential_1/lstm_4/while/lstm_cell_7/mul_2:z:0^sequential_1/lstm_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
$sequential_1/lstm_4/while/Identity_5Identity/sequential_1/lstm_4/while/lstm_cell_7/add_1:z:0^sequential_1/lstm_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_1/lstm_4/while/NoOpNoOp=^sequential_1/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp<^sequential_1/lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp>^sequential_1/lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_1_lstm_4_while_identity+sequential_1/lstm_4/while/Identity:output:0"U
$sequential_1_lstm_4_while_identity_1-sequential_1/lstm_4/while/Identity_1:output:0"U
$sequential_1_lstm_4_while_identity_2-sequential_1/lstm_4/while/Identity_2:output:0"U
$sequential_1_lstm_4_while_identity_3-sequential_1/lstm_4/while/Identity_3:output:0"U
$sequential_1_lstm_4_while_identity_4-sequential_1/lstm_4/while/Identity_4:output:0"U
$sequential_1_lstm_4_while_identity_5-sequential_1/lstm_4/while/Identity_5:output:0"
Esequential_1_lstm_4_while_lstm_cell_7_biasadd_readvariableop_resourceGsequential_1_lstm_4_while_lstm_cell_7_biasadd_readvariableop_resource_0"
Fsequential_1_lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resourceHsequential_1_lstm_4_while_lstm_cell_7_matmul_1_readvariableop_resource_0"
Dsequential_1_lstm_4_while_lstm_cell_7_matmul_readvariableop_resourceFsequential_1_lstm_4_while_lstm_cell_7_matmul_readvariableop_resource_0"
=sequential_1_lstm_4_while_sequential_1_lstm_4_strided_slice_1?sequential_1_lstm_4_while_sequential_1_lstm_4_strided_slice_1_0"ø
ysequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2|
<sequential_1/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp<sequential_1/lstm_4/while/lstm_cell_7/BiasAdd/ReadVariableOp2z
;sequential_1/lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp;sequential_1/lstm_4/while/lstm_cell_7/MatMul/ReadVariableOp2~
=sequential_1/lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp=sequential_1/lstm_4/while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°
¾
while_cond_35777
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_35777___redundant_placeholder03
/while_while_cond_35777___redundant_placeholder13
/while_while_cond_35777___redundant_placeholder23
/while_while_cond_35777___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÚI

A__inference_lstm_4_layer_call_and_return_conditional_losses_36050

inputs<
*lstm_cell_7_matmul_readvariableop_resource:>
,lstm_cell_7_matmul_1_readvariableop_resource:9
+lstm_cell_7_biasadd_readvariableop_resource:
identity¢"lstm_cell_7/BiasAdd/ReadVariableOp¢!lstm_cell_7/MatMul/ReadVariableOp¢#lstm_cell_7/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35966*
condR
while_cond_35965*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_7/BiasAdd/ReadVariableOp"^lstm_cell_7/MatMul/ReadVariableOp$^lstm_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_7/BiasAdd/ReadVariableOp"lstm_cell_7/BiasAdd/ReadVariableOp2F
!lstm_cell_7/MatMul/ReadVariableOp!lstm_cell_7/MatMul/ReadVariableOp2J
#lstm_cell_7/MatMul_1/ReadVariableOp#lstm_cell_7/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò7
ù
A__inference_lstm_3_layer_call_and_return_conditional_losses_34253

inputs#
lstm_cell_6_34171:,#
lstm_cell_6_34173:,
lstm_cell_6_34175:,
identity¢#lstm_cell_6/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskì
#lstm_cell_6/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_6_34171lstm_cell_6_34173lstm_cell_6_34175*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_34170n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¯
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_6_34171lstm_cell_6_34173lstm_cell_6_34175*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_34184*
condR
while_cond_34183*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿt
NoOpNoOp$^lstm_cell_6/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_6/StatefulPartitionedCall#lstm_cell_6/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

E
)__inference_dropout_5_layer_call_fn_39304

inputs
identity¯
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_35629`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô	
Ê
lstm_5_while_cond_37269*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1A
=lstm_5_while_lstm_5_while_cond_37269___redundant_placeholder0A
=lstm_5_while_lstm_5_while_cond_37269___redundant_placeholder1A
=lstm_5_while_lstm_5_while_cond_37269___redundant_placeholder2A
=lstm_5_while_lstm_5_while_cond_37269___redundant_placeholder3
lstm_5_while_identity
~
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: Y
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_5_while_identitylstm_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Í7
À
while_body_37643
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_6_matmul_readvariableop_resource_0:,F
4while_lstm_cell_6_matmul_1_readvariableop_resource_0:,A
3while_lstm_cell_6_biasadd_readvariableop_resource_0:,
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_6_matmul_readvariableop_resource:,D
2while_lstm_cell_6_matmul_1_readvariableop_resource:,?
1while_lstm_cell_6_biasadd_readvariableop_resource:,¢(while/lstm_cell_6/BiasAdd/ReadVariableOp¢'while/lstm_cell_6/MatMul/ReadVariableOp¢)while/lstm_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_6/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:,*
dtype0·
while/lstm_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
)while/lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:,*
dtype0
while/lstm_cell_6/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
while/lstm_cell_6/addAddV2"while/lstm_cell_6/MatMul:product:0$while/lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
(while/lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:,*
dtype0£
while/lstm_cell_6/BiasAddBiasAddwhile/lstm_cell_6/add:z:00while/lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,c
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0"while/lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_6/SigmoidSigmoid while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_6/Sigmoid_1Sigmoid while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mulMulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_6/ReluRelu while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mul_1Mulwhile/lstm_cell_6/Sigmoid:y:0$while/lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/add_1AddV2while/lstm_cell_6/mul:z:0while/lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_6/Sigmoid_2Sigmoid while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_6/Relu_1Reluwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_6/mul_2Mulwhile/lstm_cell_6/Sigmoid_2:y:0&while/lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_6/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_6/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_6/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_6/BiasAdd/ReadVariableOp(^while/lstm_cell_6/MatMul/ReadVariableOp*^while/lstm_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_6_biasadd_readvariableop_resource3while_lstm_cell_6_biasadd_readvariableop_resource_0"j
2while_lstm_cell_6_matmul_1_readvariableop_resource4while_lstm_cell_6_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_6_matmul_readvariableop_resource2while_lstm_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_6/BiasAdd/ReadVariableOp(while/lstm_cell_6/BiasAdd/ReadVariableOp2R
'while/lstm_cell_6/MatMul/ReadVariableOp'while/lstm_cell_6/MatMul/ReadVariableOp2V
)while/lstm_cell_6/MatMul_1/ReadVariableOp)while/lstm_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ç
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_38028

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

²
&__inference_lstm_5_layer_call_fn_38705
inputs_0
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_35144o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Í7
À
while_body_38286
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_7_matmul_readvariableop_resource_0:F
4while_lstm_cell_7_matmul_1_readvariableop_resource_0:A
3while_lstm_cell_7_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_7_matmul_readvariableop_resource:D
2while_lstm_cell_7_matmul_1_readvariableop_resource:?
1while_lstm_cell_7_biasadd_readvariableop_resource:¢(while/lstm_cell_7/BiasAdd/ReadVariableOp¢'while/lstm_cell_7/MatMul/ReadVariableOp¢)while/lstm_cell_7/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0·
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0£
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_7/BiasAdd/ReadVariableOp(^while/lstm_cell_7/MatMul/ReadVariableOp*^while/lstm_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_7/BiasAdd/ReadVariableOp(while/lstm_cell_7/BiasAdd/ReadVariableOp2R
'while/lstm_cell_7/MatMul/ReadVariableOp'while/lstm_cell_7/MatMul/ReadVariableOp2V
)while/lstm_cell_7/MatMul_1/ReadVariableOp)while/lstm_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
­
E
)__inference_dropout_4_layer_call_fn_38661

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_35472d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ßI

A__inference_lstm_5_layer_call_and_return_conditional_losses_35862

inputs<
*lstm_cell_8_matmul_readvariableop_resource:>
,lstm_cell_8_matmul_1_readvariableop_resource:9
+lstm_cell_8_biasadd_readvariableop_resource:
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35778*
condR
while_cond_35777*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
°
&__inference_lstm_5_layer_call_fn_38716

inputs
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_35616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
¾
while_cond_34183
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_34183___redundant_placeholder03
/while_while_cond_34183___redundant_placeholder13
/while_while_cond_34183___redundant_placeholder23
/while_while_cond_34183___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ê"
Ï
while_body_34184
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_6_34208_0:,+
while_lstm_cell_6_34210_0:,'
while_lstm_cell_6_34212_0:,
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_6_34208:,)
while_lstm_cell_6_34210:,%
while_lstm_cell_6_34212:,¢)while/lstm_cell_6/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0ª
)while/lstm_cell_6/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_6_34208_0while_lstm_cell_6_34210_0while_lstm_cell_6_34212_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_34170Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_6/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/lstm_cell_6/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity2while/lstm_cell_6/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

while/NoOpNoOp*^while/lstm_cell_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_6_34208while_lstm_cell_6_34208_0"4
while_lstm_cell_6_34210while_lstm_cell_6_34210_0"4
while_lstm_cell_6_34212while_lstm_cell_6_34212_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_6/StatefulPartitionedCall)while/lstm_cell_6/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÚI

A__inference_lstm_3_layer_call_and_return_conditional_losses_37870

inputs<
*lstm_cell_6_matmul_readvariableop_resource:,>
,lstm_cell_6_matmul_1_readvariableop_resource:,9
+lstm_cell_6_biasadd_readvariableop_resource:,
identity¢"lstm_cell_6/BiasAdd/ReadVariableOp¢!lstm_cell_6/MatMul/ReadVariableOp¢#lstm_cell_6/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_6/MatMul/ReadVariableOpReadVariableOp*lstm_cell_6_matmul_readvariableop_resource*
_output_shapes

:,*
dtype0
lstm_cell_6/MatMulMatMulstrided_slice_2:output:0)lstm_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
#lstm_cell_6/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:,*
dtype0
lstm_cell_6/MatMul_1MatMulzeros:output:0+lstm_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
lstm_cell_6/addAddV2lstm_cell_6/MatMul:product:0lstm_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,
"lstm_cell_6/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_6_biasadd_readvariableop_resource*
_output_shapes
:,*
dtype0
lstm_cell_6/BiasAddBiasAddlstm_cell_6/add:z:0*lstm_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ,]
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0lstm_cell_6/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitl
lstm_cell_6/SigmoidSigmoidlstm_cell_6/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_cell_6/mulMullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_6/ReluRelulstm_cell_6/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_6/mul_1Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_6/add_1AddV2lstm_cell_6/mul:z:0lstm_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell_6/Relu_1Relulstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_6/mul_2Mullstm_cell_6/Sigmoid_2:y:0 lstm_cell_6/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ý
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_6_matmul_readvariableop_resource,lstm_cell_6_matmul_1_readvariableop_resource+lstm_cell_6_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_37786*
condR
while_cond_37785*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_6/BiasAdd/ReadVariableOp"^lstm_cell_6/MatMul/ReadVariableOp$^lstm_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_6/BiasAdd/ReadVariableOp"lstm_cell_6/BiasAdd/ReadVariableOp2F
!lstm_cell_6/MatMul/ReadVariableOp!lstm_cell_6/MatMul/ReadVariableOp2J
#lstm_cell_6/MatMul_1/ReadVariableOp#lstm_cell_6/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
°
&__inference_lstm_3_layer_call_fn_37441

inputs
unknown:,
	unknown_0:,
	unknown_1:,
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_36238s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
ÿ
F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_34520

inputs

states
states_10
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
â


,__inference_sequential_1_layer_call_fn_35673
lstm_3_input
unknown:,
	unknown_0:,
	unknown_1:,
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCalllstm_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_35648o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_3_input
Ñ

F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_39541

inputs
states_0
states_10
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¸
serving_default¤
I
lstm_3_input9
serving_default_lstm_3_input:0ÿÿÿÿÿÿÿÿÿ;
dense_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:íû
©
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
¼
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
Ú
!cell
"
state_spec
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'_random_generator
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
¼
*	variables
+trainable_variables
,regularization_losses
-	keras_api
._random_generator
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
Ú
1cell
2
state_spec
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7_random_generator
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
¼
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>_random_generator
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
¯
Iiter

Jbeta_1

Kbeta_2
	Ldecay
Mlearning_rateAm¯Bm°Nm±Om²Pm³Qm´RmµSm¶Tm·Um¸Vm¹AvºBv»Nv¼Ov½Pv¾Qv¿RvÀSvÁTvÂUvÃVvÄ"
	optimizer
n
N0
O1
P2
Q3
R4
S5
T6
U7
V8
A9
B10"
trackable_list_wrapper
n
N0
O1
P2
Q3
R4
S5
T6
U7
V8
A9
B10"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
þ2û
,__inference_sequential_1_layer_call_fn_35673
,__inference_sequential_1_layer_call_fn_36460
,__inference_sequential_1_layer_call_fn_36487
,__inference_sequential_1_layer_call_fn_36361À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_1_layer_call_and_return_conditional_losses_36917
G__inference_sequential_1_layer_call_and_return_conditional_losses_37368
G__inference_sequential_1_layer_call_and_return_conditional_losses_36394
G__inference_sequential_1_layer_call_and_return_conditional_losses_36427À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÐBÍ
 __inference__wrapped_model_34103lstm_3_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
\serving_default"
signature_map
ø
]
state_size

Nkernel
Orecurrent_kernel
Pbias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b_random_generator
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

estates
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
û2ø
&__inference_lstm_3_layer_call_fn_37408
&__inference_lstm_3_layer_call_fn_37419
&__inference_lstm_3_layer_call_fn_37430
&__inference_lstm_3_layer_call_fn_37441Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
A__inference_lstm_3_layer_call_and_return_conditional_losses_37584
A__inference_lstm_3_layer_call_and_return_conditional_losses_37727
A__inference_lstm_3_layer_call_and_return_conditional_losses_37870
A__inference_lstm_3_layer_call_and_return_conditional_losses_38013Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_3_layer_call_fn_38018
)__inference_dropout_3_layer_call_fn_38023´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_3_layer_call_and_return_conditional_losses_38028
D__inference_dropout_3_layer_call_and_return_conditional_losses_38040´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ø
p
state_size

Qkernel
Rrecurrent_kernel
Sbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u_random_generator
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
Q0
R1
S2"
trackable_list_wrapper
5
Q0
R1
S2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

xstates
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
#	variables
$trainable_variables
%regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
û2ø
&__inference_lstm_4_layer_call_fn_38051
&__inference_lstm_4_layer_call_fn_38062
&__inference_lstm_4_layer_call_fn_38073
&__inference_lstm_4_layer_call_fn_38084Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
A__inference_lstm_4_layer_call_and_return_conditional_losses_38227
A__inference_lstm_4_layer_call_and_return_conditional_losses_38370
A__inference_lstm_4_layer_call_and_return_conditional_losses_38513
A__inference_lstm_4_layer_call_and_return_conditional_losses_38656Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_4_layer_call_fn_38661
)__inference_dropout_4_layer_call_fn_38666´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_4_layer_call_and_return_conditional_losses_38671
D__inference_dropout_4_layer_call_and_return_conditional_losses_38683´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 


state_size

Tkernel
Urecurrent_kernel
Vbias
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
T0
U1
V2"
trackable_list_wrapper
5
T0
U1
V2"
trackable_list_wrapper
 "
trackable_list_wrapper
¿
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
û2ø
&__inference_lstm_5_layer_call_fn_38694
&__inference_lstm_5_layer_call_fn_38705
&__inference_lstm_5_layer_call_fn_38716
&__inference_lstm_5_layer_call_fn_38727Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
A__inference_lstm_5_layer_call_and_return_conditional_losses_38870
A__inference_lstm_5_layer_call_and_return_conditional_losses_39013
A__inference_lstm_5_layer_call_and_return_conditional_losses_39156
A__inference_lstm_5_layer_call_and_return_conditional_losses_39299Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
:	variables
;trainable_variables
<regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_5_layer_call_fn_39304
)__inference_dropout_5_layer_call_fn_39309´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_5_layer_call_and_return_conditional_losses_39314
D__inference_dropout_5_layer_call_and_return_conditional_losses_39326´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 :2dense_1/kernel
:2dense_1/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_1_layer_call_fn_39335¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_1_layer_call_and_return_conditional_losses_39345¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
+:),2lstm_3/lstm_cell_6/kernel
5:3,2#lstm_3/lstm_cell_6/recurrent_kernel
%:#,2lstm_3/lstm_cell_6/bias
+:)2lstm_4/lstm_cell_7/kernel
5:32#lstm_4/lstm_cell_7/recurrent_kernel
%:#2lstm_4/lstm_cell_7/bias
+:)2lstm_5/lstm_cell_8/kernel
5:32#lstm_5/lstm_cell_8/recurrent_kernel
%:#2lstm_5/lstm_cell_8/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÏBÌ
#__inference_signature_wrapper_37397lstm_3_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
^	variables
_trainable_variables
`regularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_lstm_cell_6_layer_call_fn_39362
+__inference_lstm_cell_6_layer_call_fn_39379¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_39411
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_39443¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
Q0
R1
S2"
trackable_list_wrapper
5
Q0
R1
S2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
q	variables
rtrainable_variables
sregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_lstm_cell_7_layer_call_fn_39460
+__inference_lstm_cell_7_layer_call_fn_39477¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_39509
F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_39541¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
!0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
T0
U1
V2"
trackable_list_wrapper
5
T0
U1
V2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_lstm_cell_8_layer_call_fn_39558
+__inference_lstm_cell_8_layer_call_fn_39575¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_39607
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_39639¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

«total

¬count
­	variables
®	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
«0
¬1"
trackable_list_wrapper
.
­	variables"
_generic_user_object
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
0:.,2 Adam/lstm_3/lstm_cell_6/kernel/m
::8,2*Adam/lstm_3/lstm_cell_6/recurrent_kernel/m
*:(,2Adam/lstm_3/lstm_cell_6/bias/m
0:.2 Adam/lstm_4/lstm_cell_7/kernel/m
::82*Adam/lstm_4/lstm_cell_7/recurrent_kernel/m
*:(2Adam/lstm_4/lstm_cell_7/bias/m
0:.2 Adam/lstm_5/lstm_cell_8/kernel/m
::82*Adam/lstm_5/lstm_cell_8/recurrent_kernel/m
*:(2Adam/lstm_5/lstm_cell_8/bias/m
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
0:.,2 Adam/lstm_3/lstm_cell_6/kernel/v
::8,2*Adam/lstm_3/lstm_cell_6/recurrent_kernel/v
*:(,2Adam/lstm_3/lstm_cell_6/bias/v
0:.2 Adam/lstm_4/lstm_cell_7/kernel/v
::82*Adam/lstm_4/lstm_cell_7/recurrent_kernel/v
*:(2Adam/lstm_4/lstm_cell_7/bias/v
0:.2 Adam/lstm_5/lstm_cell_8/kernel/v
::82*Adam/lstm_5/lstm_cell_8/recurrent_kernel/v
*:(2Adam/lstm_5/lstm_cell_8/bias/v
 __inference__wrapped_model_34103{NOPQRSTUVAB9¢6
/¢,
*'
lstm_3_inputÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_1!
dense_1ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_1_layer_call_and_return_conditional_losses_39345\AB/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_1_layer_call_fn_39335OAB/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
D__inference_dropout_3_layer_call_and_return_conditional_losses_38028d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ¬
D__inference_dropout_3_layer_call_and_return_conditional_losses_38040d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_dropout_3_layer_call_fn_38018W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_dropout_3_layer_call_fn_38023W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¬
D__inference_dropout_4_layer_call_and_return_conditional_losses_38671d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ¬
D__inference_dropout_4_layer_call_and_return_conditional_losses_38683d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_dropout_4_layer_call_fn_38661W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_dropout_4_layer_call_fn_38666W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dropout_5_layer_call_and_return_conditional_losses_39314\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¤
D__inference_dropout_5_layer_call_and_return_conditional_losses_39326\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dropout_5_layer_call_fn_39304O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ|
)__inference_dropout_5_layer_call_fn_39309O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÐ
A__inference_lstm_3_layer_call_and_return_conditional_losses_37584NOPO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ð
A__inference_lstm_3_layer_call_and_return_conditional_losses_37727NOPO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
A__inference_lstm_3_layer_call_and_return_conditional_losses_37870qNOP?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ¶
A__inference_lstm_3_layer_call_and_return_conditional_losses_38013qNOP?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 §
&__inference_lstm_3_layer_call_fn_37408}NOPO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
&__inference_lstm_3_layer_call_fn_37419}NOPO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&__inference_lstm_3_layer_call_fn_37430dNOP?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_lstm_3_layer_call_fn_37441dNOP?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÐ
A__inference_lstm_4_layer_call_and_return_conditional_losses_38227QRSO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ð
A__inference_lstm_4_layer_call_and_return_conditional_losses_38370QRSO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
A__inference_lstm_4_layer_call_and_return_conditional_losses_38513qQRS?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ¶
A__inference_lstm_4_layer_call_and_return_conditional_losses_38656qQRS?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 §
&__inference_lstm_4_layer_call_fn_38051}QRSO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
&__inference_lstm_4_layer_call_fn_38062}QRSO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
&__inference_lstm_4_layer_call_fn_38073dQRS?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_lstm_4_layer_call_fn_38084dQRS?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÂ
A__inference_lstm_5_layer_call_and_return_conditional_losses_38870}TUVO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
A__inference_lstm_5_layer_call_and_return_conditional_losses_39013}TUVO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ²
A__inference_lstm_5_layer_call_and_return_conditional_losses_39156mTUV?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ²
A__inference_lstm_5_layer_call_and_return_conditional_losses_39299mTUV?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
&__inference_lstm_5_layer_call_fn_38694pTUVO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_lstm_5_layer_call_fn_38705pTUVO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_lstm_5_layer_call_fn_38716`TUV?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_lstm_5_layer_call_fn_38727`TUV?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÈ
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_39411ýNOP¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
 È
F__inference_lstm_cell_6_layer_call_and_return_conditional_losses_39443ýNOP¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
 
+__inference_lstm_cell_6_layer_call_fn_39362íNOP¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ
+__inference_lstm_cell_6_layer_call_fn_39379íNOP¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÈ
F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_39509ýQRS¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
 È
F__inference_lstm_cell_7_layer_call_and_return_conditional_losses_39541ýQRS¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
 
+__inference_lstm_cell_7_layer_call_fn_39460íQRS¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ
+__inference_lstm_cell_7_layer_call_fn_39477íQRS¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÈ
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_39607ýTUV¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
 È
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_39639ýTUV¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
 
+__inference_lstm_cell_8_layer_call_fn_39558íTUV¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ
+__inference_lstm_cell_8_layer_call_fn_39575íTUV¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÂ
G__inference_sequential_1_layer_call_and_return_conditional_losses_36394wNOPQRSTUVABA¢>
7¢4
*'
lstm_3_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
G__inference_sequential_1_layer_call_and_return_conditional_losses_36427wNOPQRSTUVABA¢>
7¢4
*'
lstm_3_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
G__inference_sequential_1_layer_call_and_return_conditional_losses_36917qNOPQRSTUVAB;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
G__inference_sequential_1_layer_call_and_return_conditional_losses_37368qNOPQRSTUVAB;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_sequential_1_layer_call_fn_35673jNOPQRSTUVABA¢>
7¢4
*'
lstm_3_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_1_layer_call_fn_36361jNOPQRSTUVABA¢>
7¢4
*'
lstm_3_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_1_layer_call_fn_36460dNOPQRSTUVAB;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_1_layer_call_fn_36487dNOPQRSTUVAB;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ³
#__inference_signature_wrapper_37397NOPQRSTUVABI¢F
¢ 
?ª<
:
lstm_3_input*'
lstm_3_inputÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_1!
dense_1ÿÿÿÿÿÿÿÿÿ
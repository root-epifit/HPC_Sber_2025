Основное  
<span style="color:green"> &#x2611; </span> Классические операции matmul и matvec  
<span style="color:green"> &#x2611; </span> Статическая линковка  
<span style="color:green"> &#x2611; </span> Флаги -g и -O3  
<span style="color:green"> &#x2611; </span> Вызов из python  
Дополнительное:  
<span style="color:red"> &#x2612; </span> BLAS  
<span style="color:red"> &#x2612; </span> LINPACK  
<span style="color:red"> &#x2612; </span> Штрассен  

### Заметки:

1. Если в Си вы привыкли одномерными массивами пользоваться, то можно их оставить, используя py::array_t. Я прикрепил пример, если вдруг понадобится. В cython и ctypes они тоже есть, но они все равно медленней pybind будут работать. По работе замечаний нет.

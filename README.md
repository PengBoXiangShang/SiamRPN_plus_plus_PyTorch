

<table>
   <tr>
      <td colspan = 2 rowspan=2></td>
      <td colspan = 3 style="text-align: center;">bottleneck in conv4</td>
      <td colspan = 3 style="text-align: center;">bottleneck in conv5</td>
   </tr>
   <tr>
      <td>conv1x1</td>
      <td>conv3x3</td>
      <td>conv1x1</td>
      <td>conv1x1</td>
      <td>conv3x3</td>
      <td>conv1x1</td>
   </tr>
   <tr>
      <td rowspan = 3>original ResNet-50</td>
      <td>stride</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
   </tr>
   <tr>
      <td>padding</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
   </tr>
   <tr>
      <td>dilation</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
   </tr>
   <tr>
      <td rowspan=3>ResNet-50 in SiamRPN++</td>
      <td>stride</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
   </tr>
   <tr>
      <td>padding</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
   </tr>
   <tr>
      <td>dilation</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
   </tr>
</table>

![image](https://github.com/PengBoXiangShang/tiedanernocode/blob/master/Illustrations/RPN.png)



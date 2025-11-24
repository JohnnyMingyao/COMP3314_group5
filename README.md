# COMP3314_group5
We are reproducing a LDMGI model

## Paper Information
**Title:** Image Clustering Using Local Discriminant Models and Global Integration
**Author:** Yi Yang, Dong Xu, et al
**venue:** IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 19, NO. 10, OCTOBER 2010 

## Run our code
Each part of implementation and coding are done by different group members, so the procedure to run the code is specified in each file of code. 
You can find these code in different branches

## Result
<table>
  <caption><strong>Table 2: Comparison of Original vs. Reproduced Performance (Mean ACC ± STD)</strong></caption>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">DisKmeans (DKM)</th>
      <th colspan="2">NCut</th>
      <th colspan="2">LLC-G</th>
      <th colspan="2">LDMGI</th>
    </tr>
    <tr>
      <th>Original</th>
      <th>Reproduced</th>
      <th>Original</th>
      <th>Reproduced</th>
      <th>Original</th>
      <th>Reproduced</th>
      <th>Original</th>
      <th>Reproduced</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>COIL-20</td>
      <td>53.5 ± 6.1</td>
      <td>66.9 ± 2.7</td>
      <td>68.3 ± 5.3</td>
      <td>77.4 ± 2.0</td>
      <td>67.5 ± 5.3</td>
      <td>63.5 ± 1.0</td>
      <td><strong>75.3 ± 4.9</strong></td>
      <td><strong>81.5 ± 0.5</strong></td>
    </tr>
    <tr>
      <td>USPS</td>
      <td>69.4 ± 3.7</td>
      <td>60.0 ± 0.1</td>
      <td>73.4 ± 6.3</td>
      <td>65.5 ± 3.0</td>
      <td>70.1 ± 3.9</td>
      <td>63.0 ± 1.5</td>
      <td><strong>80.5 ± 5.6</strong></td>
      <td><strong>95.7 ± 0.2</strong></td>
    </tr>
    <tr>
      <td>MNIST-T</td>
      <td>48.8 ± 3.4</td>
      <td>46.4 ± 2.6</td>
      <td>66.2 ± 3.4</td>
      <td>50.9 ± 2.0</td>
      <td>64.7 ± 3.6</td>
      <td>55.8 ± 1.0</td>
      <td><strong>71.5 ± 3.5</strong></td>
      <td><strong>78.8 ± 0.5</strong></td>
    </tr>
    <tr>
      <td>MNIST-S</td>
      <td>48.6 ± 5.6</td>
      <td>44.0 ± 1.2</td>
      <td>64.5 ± 2.0</td>
      <td>55.4 ± 1.5</td>
      <td>68.2 ± 4.5</td>
      <td>57.9 ± 1.0</td>
      <td><strong>76.3 ± 3.4</strong></td>
      <td><strong>80.0 ± 0.4</strong></td>
    </tr>
    <tr>
      <td>UMIST</td>
      <td>43.0 ± 3.2</td>
      <td>56.7 ± 3.7</td>
      <td>60.1 ± 0.7</td>
      <td>62.6 ± 1.0</td>
      <td>59.4 ± 3.2</td>
      <td>65.8 ± 0.8</td>
      <td><strong>67.0 ± 1.7</strong></td>
      <td><strong>74.6 ± 0.3</strong></td>
    </tr>
    <tr>
      <td>YALE-B</td>
      <td>41.1 ± 3.1</td>
      <td>11.0 ± 0.1</td>
      <td>46.2 ± 1.5</td>
      <td>34.2 ± 1.0</td>
      <td>46.7 ± 0.6</td>
      <td>13.1 ± 0.3</td>
      <td><strong>55.0 ± 1.2</strong></td>
      <td><strong>54.6 ± 0.5</strong></td>
    </tr>
    <tr>
      <td>JAFFE</td>
      <td>75.6 ± 9.4</td>
      <td>83.1 ± 3.3</td>
      <td>83.9 ± 6.5</td>
      <td>95.3 ± 1.6</td>
      <td>79.7 ± 5.6</td>
      <td>93.9 ± 2.0</td>
      <td><strong>90.4 ± 6.0</strong></td>
      <td><strong>92.0 ± 0.8</strong></td>
    </tr>
    <tr>
      <td>Pointing04</td>
      <td>36.3 ± 4.2</td>
      <td>48.0 ± 1.2</td>
      <td>70.6 ± 3.2</td>
      <td>45.8 ± 2.0</td>
      <td>70.2 ± 2.9</td>
      <td>61.8 ± 1.0</td>
      <td><strong>77.2 ± 1.9</strong></td>
      <td><strong>66.0 ± 0.6</strong></td>
    </tr>
    <tr>
      <td>MPEG7</td>
      <td>58.4 ± 3.9</td>
      <td>66.8 ± 1.8</td>
      <td>66.9 ± 1.8</td>
      <td>68.2 ± 1.5</td>
      <td>65.9 ± 2.0</td>
      <td>67.6 ± 0.7</td>
      <td><strong>68.0 ± 3.2</strong></td>
      <td><strong>68.0 ± 0.5</strong></td>
    </tr>
  </tbody>
</table>

<br>

<table>
  <caption><strong>Table 3: Comparison of Original vs. Reproduced Performance (Mean NMI ± STD)</strong></caption>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">DisKmeans (DKM)</th>
      <th colspan="2">NCut</th>
      <th colspan="2">LLC-G</th>
      <th colspan="2">LDMGI</th>
    </tr>
    <tr>
      <th>Original</th>
      <th>Reproduced</th>
      <th>Original</th>
      <th>Reproduced</th>
      <th>Original</th>
      <th>Reproduced</th>
      <th>Original</th>
      <th>Reproduced</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>COIL-20</td>
      <td>70.5 ± 3.2</td>
      <td>78.8 ± 1.2</td>
      <td>82.3 ± 2.4</td>
      <td>92.2 ± 1.0</td>
      <td>85.3 ± 0.5</td>
      <td>80.7 ± 0.5</td>
      <td><strong>90.0 ± 0.7</strong></td>
      <td><strong>90.3 ± 0.3</strong></td>
    </tr>
    <tr>
      <td>USPS</td>
      <td>67.6 ± 1.6</td>
      <td>58.1 ± 0.1</td>
      <td>82.4 ± 1.9</td>
      <td>61.2 ± 1.5</td>
      <td>76.8 ± 1.5</td>
      <td>74.4 ± 0.8</td>
      <td><strong>86.1 ± 2.3</strong></td>
      <td><strong>90.0 ± 1.2</strong></td>
    </tr>
    <tr>
      <td>MNIST-T</td>
      <td>45.7 ± 2.0</td>
      <td>42.0 ± 2.5</td>
      <td>67.4 ± 1.4</td>
      <td>45.4 ± 1.0</td>
      <td>62.4 ± 1.6</td>
      <td>61.9 ± 0.5</td>
      <td><strong>68.9 ± 1.5</strong></td>
      <td><strong>74.4 ± 0.3</strong></td>
    </tr>
    <tr>
      <td>MNIST-S</td>
      <td>44.8 ± 4.0</td>
      <td>39.6 ± 1.3</td>
      <td>66.6 ± 1.0</td>
      <td>47.1 ± 0.8</td>
      <td>65.1 ± 2.8</td>
      <td>63.1 ± 0.6</td>
      <td><strong>73.9 ± 1.5</strong></td>
      <td><strong>75.7 ± 0.2</strong></td>
    </tr>
    <tr>
      <td>UMIST</td>
      <td>63.6 ± 2.6</td>
      <td>77.3 ± 1.6</td>
      <td>79.1 ± 0.6</td>
      <td>75.2 ± 0.6</td>
      <td>76.2 ± 1.3</td>
      <td>81.8 ± 0.6</td>
      <td><strong>84.5 ± 1.0</strong></td>
      <td><strong>89.7 ± 0.3</strong></td>
    </tr>
    <tr>
      <td>YALE-B</td>
      <td>52.0 ± 3.1</td>
      <td>15.6 ± 1.2</td>
      <td>66.6 ± 0.6</td>
      <td>47.6 ± 0.5</td>
      <td>65.8 ± 0.1</td>
      <td>19.9 ± 0.2</td>
      <td><strong>70.8 ± 0.5</strong></td>
      <td><strong>63.5 ± 0.2</strong></td>
    </tr>
    <tr>
      <td>JAFFE</td>
      <td>79.7 ± 6.2</td>
      <td>91.8 ± 0.4</td>
      <td>90.6 ± 1.0</td>
      <td>94.2 ± 1.8</td>
      <td>89.6 ± 0.6</td>
      <td>92.4 ± 0.9</td>
      <td><strong>92.6 ± 1.4</strong></td>
      <td><strong>93.7 ± 0.5</strong></td>
    </tr>
    <tr>
      <td>Pointing04</td>
      <td>40.0 ± 4.4</td>
      <td>46.5 ± 1.0</td>
      <td>79.2 ± 1.2</td>
      <td>51.6 ± 1.1</td>
      <td>78.6 ± 1.5</td>
      <td>66.4 ± 0.5</td>
      <td><strong>84.2 ± 0.1</strong></td>
      <td><strong>77.9 ± 0.2</strong></td>
    </tr>
    <tr>
      <td>MPEG7</td>
      <td>73.7 ± 2.0</td>
      <td>79.1 ± 0.6</td>
      <td>78.5 ± 0.3</td>
      <td>79.1 ± 0.7</td>
      <td>78.0 ± 0.7</td>
      <td>78.0 ± 0.4</td>
      <td><strong>80.3 ± 0.4</strong></td>
      <td><strong>81.0 ± 0.2</strong></td>
    </tr>
  </tbody>
</table>

<!DOCTYPE html>
<html>
  <head> <meta http-equiv="Content-Type" content="text/html; charset=windows-1252">
        <title> Classify! </title>
  </head>
  <body>
    <h1> Hello World!</h1>
    <img src="img/illustration/layers.png " alt="Neuron" width="80%" height="auto" >
    <?php
                $output = exec("/var/www/kallah.co/public_html/code/python/classify.py 2>&1");
                echo $output
        ?>
  </body>
</html>


<!DOCTYPE html>
<style>
	
</style>
<html>
   <link rel="stylesheet" href="/style.css">
   <link rel="stylesheet" href="/css/font-awesome/css/font-awesome.min.css">
   <link href="https://fonts.googleapis.com/css?family=Montserrat:400,700" rel="stylesheet" type="text/css">
   <link href="https://fonts.googleapis.com/css?family=Kaushan+Script" rel="stylesheet" type="text/css">
   <link href="https://fonts.googleapis.com/css?family=Lato:400,700,400italic,700italic" rel="stylesheet" type="text/css">
   <link href="https://fonts.googleapis.com/css?family=Droid+Serif:400,700,400italic,700italic" rel="stylesheet" type="text/css">
   <link href="https://fonts.googleapis.com/css?family=Roboto+Slab:400,100,300,700" rel="stylesheet" type="text/css">
   <link rel="stylesheet" href="/kallah.css">

  <head> <meta http-equiv="Content-Type" content="text/html; charset=windows-1252">
        <title> Classify! </title>
  </head>
  <body>
<div class="col-lg-12 text-center">
	<form action="upload.php" method="post" enctype="multipart/form-data">
	    Select image to upload:
   	  <center><input type="file" name="fileToUpload" id="fileToUpload"></center>
   	  <input type="submit" value="Upload Image" name="submit">
	</form>

    <?php
		shell_exec("chmod +x /var/www/kallah.co/public_html/code/python/resized/load/*.png");
		shell_exec("chmod +x /var/www/kallah.co/public_html/code/python/resized/load/*.jpg");
		shell_exec("chmod +x /var/www/kallah.co/public_html/code/python/resized/load/*.jpeg");
		shell_exec("chmod +rmx /var/www/kallah.co/public_html/code/python/resized/load/chosen/*");
                $newline = "\n\r<br />";
                echo $newline;
		$command = "python3 /var/www/kallah.co/public_html/code/python/Classify.py 2>&1";
                $output = shell_exec($command);
		shell_exec("chmod +rmx /var/www/kallah.co/public_html/code/python/resized/load/chosen/*");
                //echo $newline;
                echo "<h1>".$output."</h1>";
		//phpinfo();
        ?>
<script src="https://ajax.aspnetcdn.com/ajax/jQuery/jquery-3.3.1.min.js"></script>
  <script>
        $(document).ready(function(){
          var folder = "resized/load/";
          var iter = 0
          $.ajax({
              url : folder,
              success: function (data) {
                  $(data).find("a").attr("href", function (i, val) {
                      if( val.match(/\.(jpe?g|png|gif)$/) ) {
                          if( iter % 5 === 0 & iter != 0){
                            $("p").append( "<br />" );
                          }
$("p").append( "<img src='"+ folder + val +"' alt='input' width='60%' height='auto'>" );

                          iter = iter + 1
                      }
                  });
              }
          });

        });
        </script>

	<p>

	</p>
</div>
  </body>
</html>

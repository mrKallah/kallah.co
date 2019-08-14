

 <?php
	ob_start();
	shell_exec("chmod +x resized/load/*.png");
	shell_exec("chmod +x resized/load/*.jpg");
	shell_exec("chmod +x resized/load/*.jpeg");
	shell_exec("chmod +rmx resized/load/chosen/*");
			$newline = "\n\r<br />";
			echo $newline;
	$command = "python3 Classify.py 2>&1 | grep 'Image is of type'";
			$output = shell_exec($command);
	shell_exec("chmod +rmx resized/load/chosen/*");
			//echo $newline;
			echo "<h1>".$output."</h1>";
	//phpinfo();
	
	file_put_contents('in/content.html', ob_get_contents());
	ob_end_flush();
	
	exec ('renew.php') ;
?>
		

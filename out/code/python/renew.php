 <?php
	
	$myfile = fopen("classify.html", "w") or die("Unable to open file!");
 
	$files = array();
	array_push($files, "in/top.html");
	
	array_push($files, "in/content.html");
	array_push($files, "in/bottom.html");
	
	foreach ($files as &$value) {
		include $value;
	}
	
	foreach ($files as &$value) {
		if ($fh = fopen($value, 'r')) {
			while (!feof($fh)) {
				$line = fgets($fh);
				fwrite($myfile, $line);
			}
			fclose($fh);
		}
	}
	
	/**/







 ?>

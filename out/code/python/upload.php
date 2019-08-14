<?php
	$name = date("Y-m-d-h-i-sa");
	$target_file = $target_dir . basename($_FILES["fileToUpload"]["name"]);
	$uploadOk = 1;
	$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));
	$ext = pathinfo($target_file, PATHINFO_EXTENSION);
	$target_dir = "resized/load/$name.$ext";


	// Check if image file is a actual image or fake image
	if(isset($_POST["submit"])) {
		 $check = getimagesize($_FILES["fileToUpload"]["tmp_name"]);
		if($check !== false) {
			// Allow certain file formats
			if($imageFileType != "jpg" && $imageFileType != "png" && $imageFileType != "jpeg") {
					echo "Sorry, only JPG, JPEG and PNG files are allowed.";
			} else {
					$remove = "rm resized/load/*.png";
					shell_exec($remove);
					$remove = "rm resized/load/*.jpg";
					shell_exec($remove);
					$remove = "rm resized/load/*.jpeg";
					shell_exec($remove);
					move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $target_dir);
					shell_exec("chmod +rmx resized/load/*.png");
					shell_exec("chmod +rmx resized/load/*.jpg");
					shell_exec("chmod +rmx resized/load/*.jpeg");
					$test = shell_exec('php classify.php');
					echo "<script>window.location.replace('classified.php');</script>";
			}
		} else {
			echo "This file is not an image or corrupt";
			#header("Location: classify.php");
			#die();
			
			
			$uploadOk = 0;
		}
	} else {
		echo "unknown error occured";
	}
?>


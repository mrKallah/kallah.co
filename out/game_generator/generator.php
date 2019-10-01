<!DOCTYPE html>
<html>

  <head>
    <link rel="shortcut icon" type="image/x-icon" href="../img/kallah.ico"></link>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>Kallah.co</title>
    <meta name="viewport" content="width=device-width">
    <meta name="description" content="This is a site set up for my bachelor that I will update on what I do, things I find interesting and generally what I think about the project.">
    <link rel="canonical" href="/">


    <!-- Custom CSS & Bootstrap Core CSS - Uses Bootswatch Flatly Theme: http://bootswatch.com/flatly/ -->
    <link rel="stylesheet" href="../style.css">


    <!-- Custom Fonts -->
    <link rel="stylesheet" href="css/font-awesome/css/font-awesome.min.css">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:400,700" rel="stylesheet" type="text/css">
    <link href='https://fonts.googleapis.com/css?family=Kaushan+Script' rel='stylesheet' type='text/css'>
    <link href="https://fonts.googleapis.com/css?family=Lato:400,700,400italic,700italic" rel="stylesheet" type="text/css">
    <link href='https://fonts.googleapis.com/css?family=Droid+Serif:400,700,400italic,700italic' rel='stylesheet' type='text/css'>
    <link href='https://fonts.googleapis.com/css?family=Roboto+Slab:400,100,300,700' rel='stylesheet' type='text/css'>

    <!-- HTML5 Shim and Respond.js IE8 support of HTML5 elements and media queries -->
    <!-- WARNING: Respond.js doesn't work if you view the page via file:// -->
    <!--[if lt IE 9]>
        <script src="https://oss.maxcdn.com/libs/html5shiv/3.7.0/html5shiv.js"></script>
        <script src="https://oss.maxcdn.com/libs/respond.js/1.4.2/respond.min.js"></script>
    <![endif]-->


</head>



  <body id="page-top" class="index">

    <!-- Navigation -->


    <!-- Header -->
    <header>
        <div class="container">
		    <nav class="navbar navbar-default ">
				<div class="container">
					<!-- Brand and toggle get grouped for better mobile display -->
					<div class="navbar-header page-scroll">
						<button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1">
							<span class="sr-only">Toggle navigation</span>
							<span class="icon-bar"></span>
							<span class="icon-bar"></span>
							<span class="icon-bar"></span>
						</button>
						<a class="navbar-brand page-scroll" href="../index.html">Kallah.co</a>
					</div>

					<!-- Collect the nav links, forms, and other content for toggling -->
					<div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
						<ul class="nav navbar-nav navbar-right">

						</ul>
					</div>
					<!-- /.navbar-collapse -->
				</div>
				<!-- /.container-fluid -->
			</nav>
            <div class="intro-text">
                <div class="intro-lead-in">Lasse Falch Sortland</div>
                <div class="intro-heading">A site about CompSci and AI</div>

            </div>
        </div>
    </header>

  <body id="page-top" class="index">


<style>
	mark.red {
		color:#c9344d;
		background: none;
	}

	mark.blue {
		color:#3466c9;
		background: none;
	}
</style>

<div class="section-subheading text-muted">
<section id="posts">
    <section id="bachelor">
      <div class="container"> 
        <div class="col-lg-12 text-center">
		<h2 id="post_title">
		<?php
			echo exec('python3 generate.py');
		?>
        </h2>
        </div>
      </div>
    </section>
  </section>
</div>
<hr />
	
    <section id="contact">
        <div class="container">
            <div class="row">
                <div class="col-lg-12 text-center">
                    <h2 class="section-heading">Contact me</h2>
					<div href="mailto:lasse_sortland@hotmail.com">
						<center>
							<a href="mailto:lasse_sortland@hotmail.com" class="btn btn-xl" style="color:black;">Mail</a>
						</center>
					</div>
                </div>
            </div>
        </div>
    </section>




      <footer>
        <div class="container">
            <div class="row">
                <div class="col-md-4">
                    <span class="copyright">Copyright &copy; Kallah.co <a" id="copyright"></a></span>
                </div>
                <div class="col-md-4">
                    <ul class="list-inline social-buttons">

                        <li><a href="https://www.facebook.com/profile.php?id=100013672805877&ref=bookmarks"><i class="fa fa-facebook"></i></a>
                        </li>

                        <li><a href="https://github.com/mrKallah"><i class="fa fa-github"></i></a>
                        </li>

                        <li><a href="https://www.youtube.com/channel/UCL09EWdEO3rK4Iu5fE2FFdA?view_as=subscriber"><i class="fa fa-youtube"></i></a>
                        </li>

                    </ul>
                </div>
                <div class="col-md-4">
                    <ul class="list-inline quicklinks">
                        <li><a href="privacy_policy">Privacy Policy</a>
                        </li>
                        <li><a href="terms_of_use">Terms of Use</a>
                        </li>
                    </ul>
                </div>

				<a class="rss-subscribe">subscribe <a href="/feed.xml">via RSS</a></a>
            </div>
        </div>
    </footer>

	<script>
	  // The current date
	  var currentDate = new Date();
	  var year = currentDate.getFullYear()

	  document.getElementById("copyright").innerHTML += year;
	</script>

     <!-- jQuery Version 1.11.0 -->
    <script src="js/jquery-1.11.0.js"></script>

    <!-- Bootstrap Core JavaScript -->
    <script src="js/bootstrap.min.js"></script>

    <!-- Plugin JavaScript -->
    <script src="js/jquery.easing.min.js"></script>
    <script src="js/classie.js"></script>
    <script src="js/cbpAnimatedHeader.js"></script>

    <!-- Contact Form JavaScript -->
    <script src="js/jqBootstrapValidation.js"></script>
    <script src="js/contact_me.js"></script>

    <!-- Custom Theme JavaScript -->
    <script src="js/agency.js"></script>


  </body>
</html>

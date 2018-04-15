<?php

DEFINE ('DB_USER', 'webdev');
DEFINE ('DB_PASSWORD', 'FunDev17');
DEFINE ('DB_HOST', 'localhost');
DEFINE ('DB_NAME', 'CS334P3');

$dbc = new mysqli(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME);
$action=$_POST["action"];

  if($action=="fillQuestions"){
     $show=$dbc->query("Select * from Questions");
      while($row=$show->fetch_assoc()){
        echo "$row[Text]#";
     }
  }
  if($action=="fillAnswers"){
     $show=$dbc->query("Select * from Content");
      while($row=$show->fetch_assoc()){
        echo "$row[Optn]*";
     }
  }
  if($action=="submitScore"){
      $user =$_POST["user"];
      $pass =$_POST["pass"];
      $scr =$_POST["score"];
      $dbc->query("insert INTO `Users` (`uID`, `uname`, `pwd`, `score`) VALUES (NULL, '$user', '$pass', '$scr')");
  }
  if($action == "getScore"){
    $show=$dbc->query("SELECT uname, score FROM Users ORDER BY score DESC");
     while($row=$show->fetch_assoc()){
       echo "$row[uname]*$row[score]#";
    }
  }
  if($action=="corrAns"){
     $show=$dbc->query("Select * from Content");
      while($row=$show->fetch_assoc()){
        echo "$row[Optn]($row[Correct]*";
     }
  }
?>

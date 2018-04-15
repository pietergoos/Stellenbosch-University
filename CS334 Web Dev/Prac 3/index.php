<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" >
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" >
    <link rel="stylesheet" href="src/main.css">
    <script type="text/javascript" language="javascript" src="src/main.js"></script>
		<script type="text/javascript" language="javascript" src="src/jquery-3.3.1.js"></script>
    <noscript>Please Update your JavaScript or enable it for the webpage to work correctly</noscript>
    <title>The Ultimate Liqour Quiz</title>
  </head>

  <body>


  </body>
</html>




<html>
   <head>

       <script type="text/javascript" src="js/jquery-1.7.1.min.js"></script>
       <script type="text/javascript" src="js/jquery-ui-1.8.17.custom.min.js"></script>

       <script type="text/javascript">
               $(document).ready(function(){

                    function showComment(){
                      $.ajax({
                        type:"post",
                        url:"process.php",
                        data:"action=showcomment",
                        success:function(data){
                             $("#comment").html(data);
                        }
                      });
                    }

                    showComment();


                    $("#button").click(function(){

                          var name=$("#name").val();
                          var message=$("#message").val();

                          $.ajax({
                              type:"post",
                              url:"process.php",
                              data:"name="+name+"&message="+message+"&action=addcomment",
                              success:function(data){
                                showComment();

                              }

                          });

                    });
               });
       </script>
   </head>

   <body>
        <form>
               name : <input type="text" name="name" id="name"/>
               </br>
               message : <input type="text" name="message" id="message" />
               </br>
               <input type="button" value="Send Comment" id="button">

               <div id="info" />
               <ul id="comment"></ul>
        </form>
   </body>
</html>

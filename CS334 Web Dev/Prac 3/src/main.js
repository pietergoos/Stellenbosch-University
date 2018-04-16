$(document).ready(function() {
  function fillQ() {
    $.ajax({
      async: false,
      type: "post",
      url: "process.php",
      data: "action=fillQuestions",
      success: function(data) {
        for (var i = 1; i < 6; i++) {
          $("#q" + i + "Text").html(data.split("#")[i - 1]);
        }
      }
    });
  }

  function fillA() {
    $.ajax({
      async: false,
      type: "post",
      url: "process.php",
      data: "action=fillAnswers",
      success: function(data) {
        var con = 1;
        var j = [0, 4, 2, 5, 6];
        for (var i = 1; i < 5; i++) {
          for (var k = 0; k < j[i]; k++) {
            $("#q" + (i + 1) + (k + 1)).html(data.split("*")[con]);
            con++;
          }
        }
      }
    });
  }
  fillQ();
  fillA();
  updtLead();

});

function submitScore(sc, us, pw) {
  $.ajax({
    async: false,
    type: "post",
    url: "process.php",
    data: "action=submitScore&user=" + us + "&pass=" + pw + "&score=" + sc,
    success: function(data) {
      $("#scrout").html("Your score was " + sc);
    }
  });
}

function reg(inp, pw) {
  var usr = /^[A-Z][a-zA-Z0-9]{4,14}/;
  var pwd = /^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[.,?!])[a-zA-Z0-9.,?!]{8,}/;
  var out = false;
  if (pw == 1) {
    out = pwd.test(inp);
  } else {
    out = usr.test(inp);
  }
  return out;
}

function updtLead() {
  $.ajax({
    async: false,
    type: "post",
    url: "process.php",
    data: "action=getScore",
    success: function(data) {

      var da = "<table class=\"table\"><thead><tr><th scope=\"col\">Place</th><th scope=\"col\">Username</th><th scope=\"col\">Score</th></tr></thead><tbody>";
      var a = data.split("#");

      for (var i = 0; i < a.length - 1; i++) {
        var u = a[i].split("*");
        da = da + "<tr><th scope=\"row\">" + (i + 1) + "</th><td>" + u[0] + "</td><td>" + u[1] + "</td></tr>";
      }
      da = da + "</tbody></table>";

      $("#tabs").html(da);
    }
  });
}

function processQ(d) {
  var score = 0;
  var comp = [1, 4, 2, 5, 6];
  var x;
  var ctr = 0;
  for (var i = 0; i < 5; i++) {
    var k = i + 1;
    for (var j = 0; j < comp[i]; j++) {
      //console.log("" + k + (j + 1));
      if (i == 0 || i == 4) {
        x = nomText(document.getElementById("" + k + (j + 1)).value);
      } else {
        x = document.getElementById("" + k + (j + 1)).checked;
      }
      var o = 1;
      var c;
      if (i == 0) {
        o = 0;
      }
      if (i != 0 && i != 4) {
        if (d[ctr].split("(")[o] == "1") {
          document.getElementById("q" + k + (j + 1)).style.backgroundColor = 'yellow';
          if (x == 1) {
            document.getElementById("q" + k + (j + 1)).style.backgroundColor = 'green';
            score++;
          }
        } else {
          if (x == 1) {
            document.getElementById("q" + k + (j + 1)).style.backgroundColor = 'red';
          }
        }
      } else if (i == 0) {
        if (d[ctr].split("(")[o] != x) {
          $("#q" + k + (j + 1)).html("The Correct answer is Whiskey");
        } else {
          $("#q" + k + (j + 1)).html("Correct!");
          score++;
        }
      } else {
        if (d[ctr].split("(")[o] != x) {
          $("#a" + k + (j + 1)).html("<i>" + d[ctr].split("(")[o] + "</i>");
        } else {
          $("#a" + k + (j + 1)).html("<i>Correct!</i>");
          score++;
        }
      }
      ctr++;
    }
  }
  /*
  var x = nomText(document.getElementById("11").value);
  if (x == "whiskey" || x == "bourbon") {
    //alert("Question 1 is correct");
    score++;
  } else {

    alert("Question 1 is incorrect: During this time Whiskey could be obtained through a prescription from your doctor and picked up at a pharmacy.");
  }

  if (document.getElementById("23").checked == true) {
    //alert("Question 2 is correct");
    score++;
  } else {
    alert("Question 2 is incorrect: Limoncello is one of Italy's national liquors.");
  }

  if (document.getElementById("31").checked == true) {
    //alert("Question 3 is correct");
    score++;
  } else {
    alert("Question 3 is incorrect: Absinthe was poured onto sugar cubes in France to improve it's taste.");
  }

  var q4ans = false;
  if (document.getElementById("41").checked == true) {
    alert("Amarula is based off of the Marula Fruit");
    q4ans = true;
  }
  if (document.getElementById("42").checked == true) {
    alert("Triple Sec is made with Oranges");
    q4ans = true;
  }
  if (document.getElementById("43").checked == false) {
    alert("Jagermeister is a spiced liquor with no fruit");
    q4ans = true;
  }
  if (document.getElementById("44").checked == true) {
    alert("Glenfiddich is made with Citrus and Pears");
    q4ans = true;
  }
  if (document.getElementById("45").checked == false) {
    alert("Sake is Japanese Rice wine, sometimes infused with fruit, but usually it is just rice");
    q4ans = true;
  }
  if (q4ans == true) {
    alert("Part of Q4 was incorrectly answered");
  } else {
    score++;
  }

  var q5ans = false;
  if (document.getElementById("51").value != "1") {
    alert("An Old Fashioned is made primarily from whiskey");
    q5ans = true;
  }
  if (document.getElementById("52").value != "2") {
    alert("A Daquiri is made primarily from Rum");
    q5ans = true;
  }
  if (document.getElementById("53").value != "2") {
    alert("A Pina Colada is made primarily from Rum");
    q5ans = true;
  }
  if (document.getElementById("54").value != "3") {
    alert("A Margarita is made primarily from Tequilla");
    q5ans = true;
  }
  if (document.getElementById("55").value != "4") {
    alert("A Lemon Drop is made primarily from Vodka");
    q5ans = true;
  }
  if (document.getElementById("56").value != "4") {
    alert("An Appletini is made primarily from Vodka");
    q5ans = true;
  }
  if (q5ans == true) {
    alert("Part of Question 5 is incorrect.");
  } else {
    score++;
  }

  alert("Your score was " + score + " out of 5");
  */
  return score;
}

function checkAll() {
  var oppp = 0;
  $.ajax({
    async: false,
    type: "post",
    url: "process.php",
    data: "action=corrAns",
    success: function(data) {
      oppp = processQ(data.split("*"));
      console.log(oppp);
    }
  });

  return oppp;
}

function nomText(string) {
  string = string.replace(/\s+/g, '');
  string = string.toLowerCase();
  return string;
}

function checkFields() {
  var empt = false;
  var ids = ["11", "51", "52", "53", "54", "55", "56"];
  for (var i = 0; i < ids.length; i++) {
    if (document.getElementById(ids[i]).value.length == 0) {
      empt = true;
      if (i == 0) {
        document.getElementById("hq1").style.backgroundColor = 'red';
      } else {
        document.getElementById("hq5").style.backgroundColor = 'red';
      }
    } else {
      if (i == 0) {
        document.getElementById("hq1").style.backgroundColor = 'white';
      } else {
        document.getElementById("hq5").style.backgroundColor = 'white';
      }
    }
  }

  if (document.getElementById("21").checked || document.getElementById("22").checked || document.getElementById("23").checked || document.getElementById("24").checked) {
    document.getElementById("hq2").style.backgroundColor = 'white';
  } else {
    empt = true;
    document.getElementById("hq2").style.backgroundColor = 'red';
  }
  if (document.getElementById("31").checked || document.getElementById("32").checked) {
    document.getElementById("hq3").style.backgroundColor = 'white';
  } else {
    empt = true;
    document.getElementById("hq3").style.backgroundColor = 'red';
  }
  if (document.getElementById("41").checked || document.getElementById("42").checked || document.getElementById("43").checked || document.getElementById("44").checked || document.getElementById("45").checked) {
    document.getElementById("hq4").style.backgroundColor = 'white';
  } else {
    empt = true;
    document.getElementById("hq4").style.backgroundColor = 'red';
  }
  if (empt) {
    alert("Not all questions have been completed!");
  }

  return !empt;
}

function subm() {
  if (checkFields()) {
    if (reg(document.getElementById("usn").value, 0)) {
      if (reg(document.getElementById("pwd").value, 0)) {
        var scr = checkAll();
        submitScore(scr, document.getElementById("usn").value, document.getElementById("pwd").value);
        updtLead();
      } else {
        alert("You must have a password with one capital, lower case, number and symbol of at least 8 characters");
      }
    } else {
      alert("You should have an alphanumeric username between 5 and 15 characters starting with a capital");
    }
  }
}

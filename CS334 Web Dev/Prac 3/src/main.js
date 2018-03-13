function checkAll() {
    var score = 0;
    var x = document.forms["q1"]["q1a"].value;
    x = nomText(x);
    if (x == "whiskey" || x == "bourbon") {
      //alert("Question 1 is correct");
      score++;
    }
    else{
      alert("Question 1 is incorrect: During this time Whiskey could be obtained through a prescription from your doctor and picked up at a pharmacy.");
    }

    if(document.getElementById("23").checked == true){
      //alert("Question 2 is correct");
      score++;
    }
    else{
      alert("Question 2 is incorrect: Limoncello is one of Italy's national liquors.");
    }

    if(document.getElementById("31").checked == true){
      //alert("Question 3 is correct");
      score++;
    }
    else{
      alert("Question 3 is incorrect: Absinthe was poured onto sugar cubes in France to improve it's taste.");
    }

    var q4ans = false;
    if(document.forms["q4"]["q4a"].checked == true){
      alert("Amarula is based off of the Marula Fruit");
      q4ans = true;
    }
    if(document.forms["q4"]["q4b"].checked == true){
      alert("Triple Sec is made with Oranges");
      q4ans = true;
    }
    if(document.forms["q4"]["q4c"].checked == false){
      alert("Jagermeister is a spiced liquor with no fruit");
      q4ans = true;
    }
    if(document.forms["q4"]["q4d"].checked == true){
      alert("Glenfiddich is made with Citrus and Pears");
      q4ans = true;
    }
    if(document.forms["q4"]["q4e"].checked == false){
      alert("Sake is Japanese Rice wine, sometimes infused with fruit, but usually it is just rice");
      q4ans = true;
    }
    if(q4ans == true){
      alert("Part of Q4 was incorrectly answered");
    }
    else{
      score++;
    }

    var q5ans = false;
    if(document.forms["q5"]["q5a"].value != "1"){
      alert("An Old Fashioned is made primarily from whiskey");
      q5ans = true;
    }
    if(document.forms["q5"]["q5b"].value != "2"){
      alert("A Daquiri is made primarily from Rum");
      q5ans = true;
    }
    if(document.forms["q5"]["q5c"].value != "2"){
      alert("A Pina Colada is made primarily from Rum");
      q5ans = true;
    }
    if(document.forms["q5"]["q5d"].value != "3"){
      alert("A Margarita is made primarily from Tequilla");
      q5ans = true;
    }
    if(document.forms["q5"]["q5e"].value != "4"){
      alert("A Lemon Drop is made primarily from Vodka");
      q5ans = true;
    }
    if(document.forms["q5"]["q5f"].value != "4"){
      alert("An Appletini is made primarily from Vodka");
      q5ans = true;
    }
    if (q5ans == true) {
      alert("Part of Question 5 is incorrect.")
    }
    else{
      score++;
    }

    alert("Your score was " + score + " out of 5")
    return false;
}

function nomText(string){
    string = string.replace(/\s+/g, '');
    string = string.toLowerCase();
    return string;
}

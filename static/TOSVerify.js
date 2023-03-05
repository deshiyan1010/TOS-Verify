function randomProcess()
{
	$.ajax({
		type: 'POST',
		url: 'randomProcess.php',
		dataType: 'json',
		success: function(data)
		{
			document.location.href = 'answers/' + data.randomstring + ".html";
		}
	});
}

function process()
{
	var text = document.getElementById('document_text').value.replace(/\r?\n|\r/g," ");
	var decodedText = text.unidecode().replace(/%/g, 'percent').replace(/&/g, 'and');
	var email = '';

	if (decodedText.length > 10000000000)
	{
		document.getElementById('toolong').style.display = 'block';
	}
	else
	{
		document.getElementById('toolong').style.display = 'none';

		$.ajax({
			type: 'POST',
			url: 'process.php',
			data: "email="+email+"&text="+decodedText,
			dataType: 'json',
			success: function(data)
			{
				document.location.href = 'thanks.php?answer=' + data.randomstring;
			}
		});
	}
}

function getURLParameter(name) {
  return decodeURIComponent((new RegExp('[?|&]' + name + '=' + '([^&;]+?)(&|#|;|$)').exec(location.search)||[,""])[1].replace(/\+/g, '%20'))||null
}

function redirectAfterTenSeconds()
{
	var randomstring = getURLParameter('answer');

	window.setTimeout(function(){

		// Move to a new location or you can do something else
        	window.location.href = "answers/" + randomstring + ".html";

	}, 10000);
}

function reloadAfterTenSeconds()
{
	window.setTimeout(function(){

		// Move to a new location or you can do something else
        	window.location.reload();

	}, 10000);
}

function toggleText()
{
	var elements = document.getElementsByClassName('not_unfair');
	for (var i = 0; i < elements.length; i++)
	{
		if (elements[i].style.display == 'block') elements[i].style.display = 'none';
		else elements[i].style.display = 'block';
	}

	var elements2 = document.getElementsByClassName('dots');
	for (var i = 0; i < elements2.length; i++)
	{
		if (elements2[i].style.display != 'none') elements2[i].style.display = 'none';
		else elements2[i].style.display = 'block';
	}
}

function share()
{
	var text = document.getElementById('results_div').innerHTML;
	var email = document.getElementById('email').value;
	var url = window.location.href;

	$.ajax({
		type: 'POST',
		url: '../share.php',
		data: "email="+email+"&text="+text+"&url="+url,
		dataType: 'json',
		success: function(data)
		{
		}
	});
}

function save()
{
	var filename = window.location.href.substring(window.location.href.lastIndexOf('/')+1).replace(".html","");

	var a = document.getElementById("save_txt");
	a.href = filename + ".txt";
	a.download = filename + ".txt";

	var a = document.getElementById("save_json");
	a.href = filename + ".json";
	a.download = filename + ".json";

	var a = document.getElementById("save_xml");
	a.href = filename + ".xml";
	a.download = filename + ".xml";
}



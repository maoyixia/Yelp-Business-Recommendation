// Author: Yixia Mao

var results;

$(document).ready(function() {
	$.getJSON('data/test.json', function(data) {
		results = data['results'];

		var displayList = [];
		var pointsSum = 0;
		for ( var i = 0; i < results.length; i++) {
			pointsSum += results[i]['cluster']['pointsNumber'];
		}

		displayList.push({
			'name' : "All Clusters",
			'pointsNum' : pointsSum,
			'index' : -1
		});

		for ( var j = 0; j < results.length; j++) {
			displayList.push({
				'name' : results[j]['cluster']['terms'][0]['text'],
				'pointsNum' : results[j]['cluster']['pointsNumber'],
				'index' : j
			});
		}

		var transform = {
			'tag' : 'option',
			'html' : '${name} (${pointsNum})',
			'value' : '${index}'
		};

		$('#displayList').json2html(displayList, transform);

		$('#displayList').change(function() {
			selectCluster(this.value);
		});

		$('#displayList').val('-1');
		$('#displayList').trigger('change');
	}).fail(function(error) {
		alert(error);
	});

});

function selectCluster(index) {
	if (index == -1) {
		var allClustersList = [];
		for ( var i = 0; i < results.length; i++) {
			var f = function(idx) {
				return function() {
					selectCluster(idx);
				}
			}
			var handler = f(i);
			allClustersList.push({
				'text' : results[i]['cluster']['terms'][0]['text'],
				'weight' : results[i]['cluster']['pointsNumber'],
				'handlers' : {
					click : handler
				}
			});
		}
		$('#wordcloud').empty();
		$('#wordcloud').jQCloud(allClustersList);
	} else {
		var list = results[index]['cluster']['terms'];
		$('#wordcloud').empty();
		$('#wordcloud').jQCloud(list);
		$('#displayList').val(index);
		$('#displayList').show();
	}
}

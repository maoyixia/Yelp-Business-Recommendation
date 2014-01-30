package visualization;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.Charset;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.clustering.AbstractClusterWriter;
import org.apache.mahout.utils.vectors.VectorHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.sf.json.JSONArray;
import net.sf.json.JSONObject;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.io.Files;

/**
 * ClusterToJSON Class is used to fetch Map of top terms string and its score
 * from clusters, and then transfer them into JSON Object.
 * 
 * @author ymao
 * 
 */
public class ClusterToJSON {

	private static final Logger log = LoggerFactory
			.getLogger(AbstractClusterWriter.class);

	public LinkedList<Pair<String, Double>> getTopFeatures(Vector vector,
			String[] dictionary, int numTerms) {

		List<TermIndexWeight> vectorTerms = Lists.newArrayList();
		Iterator<Vector.Element> iter = vector.iterateNonZero();
		while (iter.hasNext()) {
			Vector.Element elt = iter.next();
			vectorTerms.add(new TermIndexWeight(elt.index(), elt.get()));
		}

		// Sort results in reverse order (ie weight in descending order)
		Collections.sort(vectorTerms, new Comparator<TermIndexWeight>() {
			@Override
			public int compare(TermIndexWeight one, TermIndexWeight two) {
				return Double.compare(two.weight, one.weight);
			}
		});

		LinkedList<Pair<String, Double>> topTerms = new LinkedList<Pair<String, Double>>();

		for (int i = 0; i < vectorTerms.size() && i < numTerms; i++) {
			int index = vectorTerms.get(i).index;
			String dictTerm = dictionary[index];
			if (dictTerm == null) {
				log.error("Dictionary entry missing for {}", index);
				continue;
			}
			topTerms.add(new Pair<String, Double>(dictTerm,
					vectorTerms.get(i).weight));
		}
		return topTerms;
	}

	public JSONObject topFeaturesToJSON(
			LinkedList<Pair<String, Double>> topTerms, double pointsNum) {
		JSONArray termsArray = new JSONArray();
		JSONObject termsObject = new JSONObject();
		JSONObject clusterObject = new JSONObject();
		JSONObject clustersObject = new JSONObject();

		for (Pair<String, Double> term : topTerms) {
			termsObject.put("text", term.getFirst());
			termsObject.put("weight", term.getSecond());
			termsArray.add(termsObject);
		}
		clusterObject.put("terms", termsArray);
		clusterObject.put("pointsNumber", pointsNum);

		clustersObject.put("cluster", clusterObject);
		return clustersObject;
	}

	private static class TermIndexWeight {
		private final int index;
		private final double weight;

		TermIndexWeight(int index, double weight) {
			this.index = index;
			this.weight = weight;
		}
	}

	public static void main(String[] args) throws IOException {

		Configuration conf = new Configuration();
		String termDictionary = "results/resumes/resumes-clusters6/dictionary.file-0";
		String[] dictionary = VectorHelper.loadTermDictionary(conf,
				termDictionary);

		String seqFile = "results/resumes/resumes-clusters6/clusters/clusters-100-final/part-r-00000";
		Path seqFilePath = new Path(seqFile);

		/* choose your number of top terms per cluster here */
		int termsNum = 30;

		JSONObject resultObject = new JSONObject();
		JSONArray resultArray = new JSONArray();

		SequenceFileDirValueIterable<ClusterWritable> seqFileIter = new SequenceFileDirValueIterable<ClusterWritable>(
				seqFilePath, PathType.GLOB, conf);
		Iterator<ClusterWritable> iter = seqFileIter.iterator();

		while (iter.hasNext()) {
			ClusterWritable clusterWritable = iter.next();
			Cluster cluster = clusterWritable.getValue();
			ClusterToJSON clusterToJSON = new ClusterToJSON();
			LinkedList<Pair<String, Double>> topTerms = clusterToJSON
					.getTopFeatures(cluster.getCenter(), dictionary, termsNum);
			double pointsNum = cluster.getNumObservations();
			System.out.println("Points Number: " + pointsNum + "\n");

			for (Pair<String, Double> topterm : topTerms) {
				System.out.println("term: " + topterm.getFirst()
						+ "\n\t\t score: " + topterm.getSecond());
			}

			resultArray.add(clusterToJSON
					.topFeaturesToJSON(topTerms, pointsNum));
		}

		resultObject.put("results", resultArray);
		FileUtils.writeStringToFile(new File("test.json"),
				resultObject.toString());
	}
}

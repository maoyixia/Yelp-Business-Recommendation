package clustering;

import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.common.distance.TanimotoDistanceMeasure;
import org.apache.mahout.math.hadoop.stats.BasicStats;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.HighDFWordsPruner;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;

/**
 * ClusteringDriver is used for clustering input documents in sequencefile
 * format. It supports Distance Measure: EUCLIDEAN, SQUARED_EUCLIDEAN,
 * MANHATTAN, COSINE and TANIMOTO; Centroids generate algorithm: RANDOM and
 * CANOPY; Clustering algorithm: KMEANS and FUZZYKMEANS.
 * 
 * @author ymao
 * 
 */
public class ClusteringDriver {

	private enum DistanceMeasureType {
		EUCLIDEAN, SQUARED_EUCLIDEAN, MANHATTAN, COSINE, TANIMOTO
	}

	private enum CentroidsAlgo {
		RANDOM, CANOPY
	}

	private enum ClusteringAlgo {
		KMEANS, FUZZYKMEANS
	}

	public static void main(String args[]) throws Exception {

		// parameters
		int minSupport = 2;
		int minDf = 2;
		int maxDFPercent = 90;
		int maxNGramSize = 2;
		int minLLRValue = 50;
		int reduceTasks = 1;
		int chunkSize = 200;
		int norm = 2;
		boolean sequentialAccessOutput = true;
		boolean namedVectors = true;

		/* choose your number of clustering iterations here */
		int clusteringAlgoIterationNum = 100;

		/*
		 * if use Canopy to generate centroids automatically, choose your canopy
		 * T1 and T2 here
		 */
		double T1 = 0.5;
		double T2 = 0.5;
		/*
		 * If not use Canopy to generate centroids automatically, choose your
		 * number of clusters here
		 */
		int clusterNum = 20;

		/* choose your centroids generated algorithm here */
		CentroidsAlgo centroidsAlgo = CentroidsAlgo.RANDOM;
		/* choose your clustering algorithm here */
		ClusteringAlgo clusteringAlgo = ClusteringAlgo.KMEANS;
		/* choose your distance measure here */
		DistanceMeasureType distanceMeasureType = DistanceMeasureType.COSINE;

		DistanceMeasure distanceMeasure = null;
		switch (distanceMeasureType) {
		case EUCLIDEAN:
			distanceMeasure = new EuclideanDistanceMeasure();
			break;
		case SQUARED_EUCLIDEAN:
			distanceMeasure = new SquaredEuclideanDistanceMeasure();
			break;
		case MANHATTAN:
			distanceMeasure = new ManhattanDistanceMeasure();
			break;
		case COSINE:
			distanceMeasure = new CosineDistanceMeasure();
			break;
		case TANIMOTO:
			distanceMeasure = new TanimotoDistanceMeasure();
			break;
		}

		// directory of doc sequence file(s)
		String inputDirName = args[0];
		Path inputDir = new Path(inputDirName);

		// directory where clusters will be written
		String outputDirName = args[1];
		Path outputDir = new Path(outputDirName);

		Configuration conf = new Configuration();

		HadoopUtil.delete(conf, outputDir);

		// converts input docs in sequence file format in input_dir into token
		// array in output_dir/tokenized-documents
		Path tokenizedPath = new Path(outputDirName,
				DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
		MyAnalyzer analyzer = new MyAnalyzer();
		DocumentProcessor.tokenizeDocuments(inputDir, analyzer.getClass()
				.asSubclass(Analyzer.class), tokenizedPath, conf);

		// reads token array in output_dir/tokenized-documents and writes term
		// frequency vectors in output_dir (under tf-vectors)
		String tfDirName = DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER;
		DictionaryVectorizer.createTermFrequencyVectors(tokenizedPath,
				outputDir, tfDirName, conf, minSupport, maxNGramSize,
				minLLRValue, -1.0f, false, reduceTasks, chunkSize,
				sequentialAccessOutput, namedVectors);

		// converts term frequency vectors in output_dir/tf-vectors to TF-IDF
		// vectors in output_dir (under tfidf-vectors)
		Pair<Long[], List<Path>> calculateDF = TFIDFConverter.calculateDF(
				new Path(outputDir, tfDirName), outputDir, conf, chunkSize);
		long maxDF = maxDFPercent;
		TFIDFConverter.processTfIdf(new Path(outputDir,
				DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER), outputDir,
				conf, calculateDF, minDf, maxDF, norm, true,
				sequentialAccessOutput, namedVectors, reduceTasks);

		// reads tfidf-vectors from output_dir/tfidf-vectors and writes out
		// centroids at output_dir/centroids
		Path tfidfVectors = new Path(outputDirName, "tfidf-vectors");
		Path clusterOutput = new Path(outputDirName, "clusters");
		Path centroids = new Path(outputDirName, "centroids");
		if (centroidsAlgo == CentroidsAlgo.RANDOM) {
			RandomSeedGenerator.buildRandom(conf, tfidfVectors, centroids,
					clusterNum, distanceMeasure);
		} else if (centroidsAlgo == CentroidsAlgo.CANOPY) {
			CanopyDriver.run(conf, tfidfVectors, centroids, distanceMeasure,
					T1, T2, false, 0.01, false);
			centroids = new Path(centroids, "clusters-0-final");
		} else {
			System.err.println("Please choose centroids generate algorithm.");
		}

		// reads tfidf-vectors from output_dir/tfidf-vectors and refers to
		// directory path for initial clusters, and
		// writes out clusters to output_dir/clusters
		if (clusteringAlgo == ClusteringAlgo.KMEANS) {
			KMeansDriver.run(conf, tfidfVectors, centroids, clusterOutput,
					distanceMeasure, 0.01, clusteringAlgoIterationNum, true,
					0.01, false);
		} else if (clusteringAlgo == ClusteringAlgo.FUZZYKMEANS) {
			FuzzyKMeansDriver.run(conf, tfidfVectors, centroids, clusterOutput,
					distanceMeasure, 0.01, clusteringAlgoIterationNum, 2.0f,
					true, true, 0.0, false);
		} else {
			System.err.println("Please choose clustering algorithm.");
		}
	}
}

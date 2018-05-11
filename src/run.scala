import breeze.linalg.DenseVector
import org.apache.spark._
import org.apache.log4j._
import org.apache.spark.sql.{Row, SparkSession}



object run {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    val sc = new SparkContext("local[*]", "Perceptron")


    val sparkSession = SparkSession.builder
      .master("local")
      .appName("my-spark-app")
//      .config("spark.some.config.option", "config-value")
      .getOrCreate()

    val data_csv = sparkSession.read
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .load("./data/mult-class_data.csv")


    val acc = (1 to 20).map(_=>{

      val features = data_csv.drop("_c4").rdd.map(row => {
        val f_DV = DenseVector(row.toSeq.toArray.map(x=> x.asInstanceOf[String].toDouble))
        val v = DenseVector.ones[Double](f_DV.length+1 )
        v(1 to f_DV.length):=f_DV
        v
      })
      val labels = data_csv.select("_c4").rdd.map(r => r(0).asInstanceOf[String].toDouble)

      val Array(train_data,test_data) = features.zip(labels).randomSplit(Array(0.7, 0.3))
      val train_features = train_data.map(x=>x._1)
      val train_label = train_data.map(x=>x._2)
      val test_features = test_data.map(x=>x._1)
      val test_labels = test_data.map(x=>x._2)


      val act_f = new PerceptronActFunction
      val model: MultiClassPerceptron = new MultiClassPerceptron(0.002f,500, act_f.sigmoid)
      model.fit(train_features, train_label)

      val X_y = test_features.zip(test_labels)
      val accuracy = X_y.map(data => {
        if (Math.abs(model.prediction(data._1) - data._2) <0.01) 1.0 else 0.0
      }).reduce((x,y) => x + y) / test_features.count() * 100
      println(f"Accuracy on test data:  $accuracy%2.2f%%")

      accuracy

    })

    acc.foreach(println)

  }

}

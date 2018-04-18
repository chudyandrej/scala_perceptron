package org.apache



import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix, DenseVector, csvread}
import breeze.numerics.exp
import org.apache.spark.SparkContext
import org.apache.spark.util.AccumulatorV2


class Perceptron(var eta: Double, var n_iter: Int) extends java.io.Serializable  {




  def fit(X: RDD[DenseVector[Double]], y: RDD[Int]): Unit = {

    val learning_rate = eta
    var w = DenseVector.zeros[Double](X.first().length)

    val X_y = X.zip(y).cache()

    for(_ <- 1 to n_iter) {
      println(w)
      val delta_w = X_y.map(data => {
        val pred = prediction(data._1, w)
        learning_rate * (data._2 - pred) * data._1
      }).collect()

      delta_w.foreach(println)

      val no_null_delta_w = delta_w.filter(vector => vector != DenseVector.zeros[Double](vector.length))
      val no_null_delta_w_size = no_null_delta_w.length
      val new_w = no_null_delta_w.reduce((x, y) => x + y) * (1 / no_null_delta_w_size.toDouble)
      w += new_w


    }

  }


  def prediction(features: DenseVector[Double], w:DenseVector[Double]): Int = {
    if (1 / 1 + exp(-(w.t * features)) >= 0) 0 else 1
  }
}
package org.apache



import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseVector, csvread}
import breeze.numerics.exp
import org.apache.spark.SparkContext
import org.apache.spark.util.AccumulatorV2



class Perceptron(var l_rate: Double, var n_epoch: Int)  extends java.io.Serializable {

  var w : DenseVector[Double] = _


  def fit(X:RDD[DenseVector[Double]], y:RDD[Int], sc:SparkContext): Unit = {
    val myVectorAcc = new VectorAccumulatorV2(X.first().length)
    sc.register(myVectorAcc, "MyVectorAcc1")

    val learning_rate = l_rate
    var w_local = DenseVector.zeros[Double](X.first().length)
    val X_y = X.zip(y)

    for( _ <- 1 to n_epoch){

      val delta_w = X_y.map(f_l => {
        val x = myVectorAcc.value.t * f_l._1
        if (1 / 1 + exp(-x) >= 0){
          myVectorAcc.add(f_l._1 * (learning_rate * (f_l._2 - 0)))

        } else {
          myVectorAcc.add(f_l._1 * (learning_rate * (f_l._2 - 1)))

        }
      })
     // delta_w.foreach(println)


    }

    this.w = w_local

//    println(this.w)
  }


  def predicate(X:DenseVector[Double]): Int ={
    val tmp = this.w.t * X
    if(1 / 1 + exp(-tmp) >= 0){
      0
    } else {
      1
    }
  }


}





class VectorAccumulatorV2(var size:Int) extends AccumulatorV2[DenseVector[Double], DenseVector[Double]] {

  private var myVector: DenseVector[Double] = DenseVector.zeros[Double](size)

  def reset(): Unit = {
    myVector =  DenseVector.zeros[Double](size)
  }

  def add(v: DenseVector[Double]): Unit = {
    myVector += v
  }

  override def isZero: Boolean = ???

  override def copy(): AccumulatorV2[DenseVector[Double], DenseVector[Double]] = ???

  override def merge(other: AccumulatorV2[DenseVector[Double], DenseVector[Double]]): Unit = ???

  override def value: DenseVector[Double] = myVector
}


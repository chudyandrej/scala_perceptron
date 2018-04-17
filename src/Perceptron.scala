package org.apache



import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix, DenseVector, csvread}
import breeze.numerics.exp
import org.apache.spark.SparkContext
import org.apache.spark.util.AccumulatorV2



class Perceptron(var l_rate: Double, var n_epoch: Int = 100)  extends java.io.Serializable {



  def fit(X:RDD[DenseVector[Double]], y:RDD[Int], sc:SparkContext): Unit = {

    val myVectorAcc =  VectorAccumulatorV2
    myVectorAcc.init(X.first().length)
    sc.register(myVectorAcc, "MyVectorAcc1")
    val counter = sc.longAccumulator("counter")

    val learning_rate = l_rate
    val X_y = X.zip(y)

    for( _ <- X_y){


      myVectorAcc.add(DenseVector(0,0,0,1))
      counter.add(1)

      //      X_y.foreach(f_l => {
      ////        f_l._1 * (0.01 * (f_l._2.toInt - 0)))
      //
      //
      //
      //
      //      })
    }



    println(counter.value)
    println(myVectorAcc.value)

  }



}





object VectorAccumulatorV2 extends AccumulatorV2[DenseVector[Double], DenseVector[Double]] {

  private var size: Int = 1
  private var myVector: DenseVector[Double] = DenseVector.zeros[Double](size)


  def init(size: Int): Unit = {
    this.size = size
  }

  def reset(): Unit = {
    myVector =  DenseVector.zeros[Double](size)
  }

  def add(v: DenseVector[Double]): Unit = {
    myVector += v
  }

  override def isZero: Boolean = myVector == DenseVector.zeros[Double](size)

  override def copy(): AccumulatorV2[DenseVector[Double], DenseVector[Double]] = {
    VectorAccumulatorV2

  }

  override def value: DenseVector[Double] = myVector

  override def merge(other: AccumulatorV2[DenseVector[Double], DenseVector[Double]]): Unit = {
    DenseVector.vertcat(this.myVector,other.value)

  }
}
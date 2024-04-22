using System;

public class Particle
{
    public double[] position;
    public double[] velocity;
    public double[] pbest;
    public double error;
    public double ebest;

    public Particle(double[] position, double[] velocity, double[] pbest, double error, double ebest)
    {
        this.position = new double[position.Length];
        position.CopyTo(this.position, 0);
        this.velocity = new double[velocity.Length];
        velocity.CopyTo(this.velocity, 0);
        this.pbest = new double[pbest.Length];
        pbest.CopyTo(this.pbest, 0);
        this.error = error;
        this.ebest = ebest;
    }
}
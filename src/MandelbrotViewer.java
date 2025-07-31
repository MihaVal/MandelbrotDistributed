import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import mpi.*;

public class MandelbrotViewer extends JFrame {
    // State Fields
    private int width = 800;
    private int height = 600;
    private static final int MAX_ITER = 250;
    private double xMin = -2.0, xMax = 1.0;
    private double yMin = -1.2, yMax = 1.2;

    // Control Flags (volatile for thread safety)
    private volatile boolean renderRequested = true; // Request initial render
    private volatile boolean shutdownRequested = false;

    // GUI and MPI Fields
    private BufferedImage image;
    private final int rank;
    private final int size;

    public MandelbrotViewer(int rank, int size) {
        this.rank = rank;
        this.size = size;
        if (rank == 0) {
            this.image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            initGUI();
        }
    }

    private void initGUI() {
        setTitle("Mandelbrot MPI Viewer (Master)");
        setSize(width, height);
        setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        setLocationRelativeTo(null);

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                requestShutdown();
            }
        });

        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent ev) {
                if (renderingIsInProgress()) return; // Don't accept input during render

                double panSpeed = 0.1 * (xMax - xMin);
                switch (ev.getKeyCode()) {
                    case KeyEvent.VK_LEFT:  xMin -= panSpeed; xMax -= panSpeed; break;
                    case KeyEvent.VK_RIGHT: xMin += panSpeed; xMax += panSpeed; break;
                    case KeyEvent.VK_UP:    yMin -= panSpeed; yMax -= panSpeed; break;
                    case KeyEvent.VK_DOWN:  yMin += panSpeed; yMax += panSpeed; break;
                    case KeyEvent.VK_1:     zoom(0.8); break;
                    case KeyEvent.VK_2:     zoom(1.25); break;
                    case KeyEvent.VK_Q:     requestShutdown(); return;
                }
                requestRender();
            }
        });

        addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                if (renderingIsInProgress()) return;
                width = getWidth();
                height = getHeight();
                if (width > 0 && height > 0) {
                    image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                    requestRender();
                }
            }
        });
    }

    private void zoom(double factor) {
        double xCenter = (xMin + xMax) / 2;
        double yCenter = (yMin + yMax) / 2;
        double xRange = (xMax - xMin) * factor;
        double yRange = (yMax - yMin) * factor;
        xMin = xCenter - xRange / 2; xMax = xCenter + xRange / 2;
        yMin = yCenter - yRange / 2; yMax = yCenter + yRange / 2;
    }

    // Thread-safe methods for main thread to control state
    public synchronized void requestRender() { this.renderRequested = true; }
    public synchronized void requestShutdown() { this.shutdownRequested = true; }
    private synchronized boolean isRenderRequested() { return this.renderRequested; }
    private synchronized boolean isShutdownRequested() { return this.shutdownRequested; }
    private synchronized boolean renderingIsInProgress() { return this.renderRequested; }
    private synchronized void setRenderDone() { this.renderRequested = false; }



    public void runMasterLoop() {
        while (!isShutdownRequested()) {
            if (isRenderRequested()) {
                try {
                    // MPI render process happens here
                    System.out.printf("Master thread starting render for R:[%.4f, %.4f]\n", xMin, xMax);

                    double[] params = new double[]{width, height, xMin, xMax, yMin, yMax};
                    MPI.COMM_WORLD.Bcast(params, 0, 6, MPI.DOUBLE, 0);

                    int rowsPerProc = height / size;
                    int extraRows = height % size;
                    int yStart = rank * rowsPerProc + Math.min(rank, extraRows);
                    int yEnd = yStart + rowsPerProc + (rank < extraRows ? 1 : 0);
                    int[] localPixels = computeSlice(width, height, xMin, xMax, yMin, yMax, yStart, yEnd);

                    int[] recvCounts = new int[size];
                    int[] displs = new int[size];
                    int offset = 0;
                    for (int i = 0; i < size; i++) {
                        int rpp = height / size;
                        int er = height % size;
                        int startRow = i * rpp + Math.min(i, er);
                        int endRow = startRow + rpp + (i < er ? 1 : 0);
                        recvCounts[i] = (endRow - startRow) * width;
                        displs[i] = offset;
                        offset += recvCounts[i];
                    }

                    int[] allPixels = new int[width * height];
                    MPI.COMM_WORLD.Gatherv(localPixels, 0, localPixels.length, MPI.INT,
                            allPixels, 0, recvCounts, displs, MPI.INT, 0);

                    // --- Safely update the GUI on the EDT ---
                    SwingUtilities.invokeLater(() -> {
                        image.setRGB(0, 0, width, height, allPixels, 0, width);
                        repaint();
                        setRenderDone(); // Mark render as complete
                    });

                } catch (Exception e) {
                    e.printStackTrace();
                    setRenderDone();
                }
            }
            try { Thread.sleep(10); } catch (InterruptedException e) {} // Small sleep to prevent busy-waiting
        }

        // Shutdown sequence
        try {
            System.out.println("Master sending shutdown signal...");
            double[] quitSignal = new double[]{-1};
            MPI.COMM_WORLD.Bcast(quitSignal, 0, 1, MPI.DOUBLE, 0);
        } catch(MPIException e) {
            e.printStackTrace();
        }

        dispose(); // Close the GUI window
    }


    public void runWorkerLoop() {
        while (true) {
            try {
                double[] params = new double[6];
                MPI.COMM_WORLD.Bcast(params, 0, 6, MPI.DOUBLE, 0);

                if (params[0] < 0) { // Shutdown signal
                    System.out.println("Worker " + rank + " received shutdown. Exiting.");
                    break;
                }

                int w = (int) params[0], h = (int) params[1];
                double xm = params[2], xM = params[3], ym = params[4], yM = params[5];

                int rowsPerProc = h / size;
                int extraRows = h % size;
                int yStart = rank * rowsPerProc + Math.min(rank, extraRows);
                int yEnd = yStart + rowsPerProc + (rank < extraRows ? 1 : 0);

                int[] localPixels = computeSlice(w, h, xm, xM, ym, yM, yStart, yEnd);

                MPI.COMM_WORLD.Gatherv(localPixels, 0, localPixels.length, MPI.INT, null, 0, null, null, MPI.INT, 0);
            } catch (MPIException e) {
                e.printStackTrace();
                break;
            }
        }
    }

    @Override
    public void paint(Graphics g) {
        super.paint(g);
        if (image != null) {
            g.drawImage(image, 0, 0, getWidth(), getHeight(), null);
        }
    }

    // computation logic
    private static int[] computeSlice(int w, int h, double xm, double xM, double ym, double yM, int yStart, int yEnd) {
        if (yStart >= yEnd || w <= 0) return new int[0];
        int[] slicePixels = new int[(yEnd - yStart) * w];
        int idx = 0;
        for (int y = yStart; y < yEnd; y++) {
            for (int x = 0; x < w; x++) {
                double real = xm + x * (xM - xm) / (w - 1);
                double imag = ym + y * (yM - ym) / (h - 1);
                Complex c = new Complex(real, imag);
                Complex z = new Complex(0, 0);
                int n = 0;
                while (z.absSq() <= 4.0 && n < MAX_ITER) {
                    z = z.multiply(z).add(c);
                    n++;
                }
                if (n == MAX_ITER) {
                    slicePixels[idx++] = Color.BLACK.getRGB();
                } else {
                    slicePixels[idx++] = Color.getHSBColor(0.7f + (float) n / MAX_ITER, 1.0f, 1.0f).getRGB();
                }
            }
        }
        return slicePixels;
    }

    public static void main(String[] args) {
        try {
            MPI.Init(args);
            int rank = MPI.COMM_WORLD.Rank();
            int size = MPI.COMM_WORLD.Size();

            if (rank == 0) {
                // Master Process
                MandelbrotViewer masterViewer = new MandelbrotViewer(rank, size);
                SwingUtilities.invokeLater(() -> masterViewer.setVisible(true));
                masterViewer.runMasterLoop(); // The main thread now enters the consumer loop

            } else {
                // Worker Process
                MandelbrotViewer workerViewer = new MandelbrotViewer(rank, size);
                workerViewer.runWorkerLoop();
            }

            MPI.Finalize(); // All processes will reach here upon graceful shutdown
            System.out.println("Process " + rank + " finalized.");

        } catch (MPIException e) {
            e.printStackTrace();
        }
    }

    static class Complex {
        private final double real, imag;
        public Complex(double r, double i) { real = r; imag = i; }
        public Complex add(Complex o) { return new Complex(real + o.real, imag + o.imag); }
        public Complex multiply(Complex o) { return new Complex(real * o.real - imag * o.imag, real * o.imag + imag * o.real); }
        public double absSq() { return real * real + imag * imag; }
    }
}
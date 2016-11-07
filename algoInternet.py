static long distSq(Point a, Point b) {
    return ((long) (a.x - b.x) * (long) (a.x - b.x) + (long) (a.y - b.y) * (long) (a.y - b.y));
}

static long ccw(Point p1, Point p2, Point p3) {
    return (long) (p2.x - p1.x) * (long) (p3.y - p1.y) - (long) (p2.y - p1.y) * (long) (p3.x - p1.x);
}

static List<Point> convexHull(List<Point> P) {

    if (P.size() < 3) {
        //WTF
        return null;
    }

    int k = 0;

    for (int i = 0; i < P.size(); i++) {
        if (P.get(i).y < P.get(k).y || (P.get(i).y == P.get(k).y && P.get(i).x < P.get(k).x)) {
            k = i;
        }
    }

    Collections.swap(P, k, P.size() - 1);

    final Point o = P.get(P.size() - 1);
    P.remove(P.size() - 1);


    Collections.sort(P, new Comparator() {

        public int compare(Object o1, Object o2) {
            Point a = (Point) o1;
            Point b = (Point) o2;

            long t1 = (long) (a.y - o.y) * (long) (b.x - o.x) - (long) (a.x - o.x) * (long) (b.y - o.y);

            if (t1 == 0) {
                long tt = distSq(o, a);
                tt -= distSq(o, b);
                if (tt > 0) {
                    return 1;
                } else if (tt < 0) {
                    return -1;
                }
                return 0;

            }
            if (t1 < 0) {
                return -1;
            }
            return 1;

        }
    });



    List<Point> hull = new ArrayList<Point>();
    hull.add(o);
    hull.add(P.get(0));


    for (int i = 1; i < P.size(); i++) {
        while (hull.size() >= 2 &&
                ccw(hull.get(hull.size() - 2), hull.get(hull.size() - 1), P.get(i)) <= 0) {
            hull.remove(hull.size() - 1);
        }
        hull.add(P.get(i));
    }

    return hull;
}

static long nearestPoints(List<Point> P, int l, int r) {


    if (r - l == P.size()) {

        Collections.sort(P, new Comparator() {

            public int compare(Object o1, Object o2) {
                int t = ((Point) o1).x - ((Point) o2).x;
                if (t == 0) {
                    return ((Point) o1).y - ((Point) o2).y;
                }
                return t;
            }
        });
    }

    if (r - l <= 100) {
        long ret = distSq(P.get(l), P.get(l + 1));
        for (int i = l; i < r; i++) {
            for (int j = i + 1; j < r; j++) {
                ret = Math.min(ret, distSq(P.get(i), P.get(j)));
            }
        }
        return ret;

    }

    int c = (l + r) / 2;
    long lD = nearestPoints(P, l, c);
    long lR = nearestPoints(P, c + 1, r);
    long ret = Math.min(lD, lR);
    Set<Point> set = new TreeSet<Point>(new Comparator<Point>() {

        public int compare(Point o1, Point o2) {
            int t = o1.y - o2.y;
            if (t == 0) {
                return o1.x - o2.x;
            }
            return t;
        }
    });
    for (int i = l; i < r; i++) {
        set.add(P.get(i));
    }

    int x = P.get(c).x;

    double theta = Math.sqrt(ret);

    Point[] Q = set.toArray(new Point[0]);
    Point[] T = new Point[Q.length];
    int pos = 0;
    for (int i = 0; i < Q.length; i++) {
        if (Q[i].x - x + 1 > theta) {
            continue;
        }
        T[pos++] = Q[i];
    }

    for (int i = 0; i < pos; i++) {
        for (int j = 1; j < 7 && i + j < pos; j++) {
            ret = Math.min(ret, distSq(T[i], T[j + i]));
        }
    }
    return ret;
}
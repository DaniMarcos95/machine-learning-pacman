import re,csv,sys

print(sys.argv)
filename = sys.argv[1]
outfilename = sys.argv[2]

outfile = csv.writer(open(outfilename,'w'),lineterminator='\n')
outfile.writerow(['num','steps','steps_t','t','r','e','Q','won'])


with open(filename,'r') as f:
    for line in f.readlines():
        print(line.rstrip('\n'))
        vals = re.findall('#\s+(\d+)\s\|\ssteps:\s+(\d+)\s\|\ssteps_t:\s+(\d+)\s\|\st:\s+(\d+\.\d+)\s\|\sr:\s+(-?\d+\.\d+)\s\|\se:\s+(-?\d+\.\d+)\s\|\sQ:\s+(\S+)\s\|\swon:\s(\w+)',line)
        if vals == []:
            print("NO MATCH ON LINE: " + line)
        else:
            print(vals[0])
            outfile.writerow(vals[0])
            
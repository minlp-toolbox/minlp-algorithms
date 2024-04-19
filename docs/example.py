from benders_exp.quick_and_dirty import Solver, MinlpData, MinlpProblem

problem = MinlpProblem()
data = MinlpData()

s = Solver("benders", problem, data)

result = s.solve(data)
# Alter data
data.lbx = [....]
s.solve(data)
s.get_stats().save("myfile.pickle")
data.save("results.pickle")
data.load("results.pickle")

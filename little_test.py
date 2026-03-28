# visited = [False for _ in range(5)]
ans = []
def treverse(visited, sum, path):
    for i in range(5):
        if not visited[i]:
            visited[i] = True
            print(i)
            if i == 0:
                sum /= 2
            elif i == 1:
                sum -= 900
            elif i == 2:
                sum += 2000
            elif i == 3:
                sum *= 5
            else:
                sum += 500
            if sum == 3000:
                ans.append(path + [i])
            
            treverse(visited, sum, path + [i])

            visited[i] = False
            if i == 0:
                sum *= 2
            elif i == 1:
                sum += 900
            elif i == 2:
                sum -= 2000
            elif i == 3:
                sum /= 5
            else:
                sum -= 500

visit = [False for _ in range(5)]
treverse(visit, 0, [])
print(ans)
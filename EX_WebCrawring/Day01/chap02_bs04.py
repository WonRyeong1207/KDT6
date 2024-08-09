from bs4 import BeautifulSoup

tr = '''
<table>
    <tr class="passed a b c" id="row1 example"><td>t1</td></tr>
    <tr class="failed" id="row2"><td>t2</td></tr>
</table>
'''

table = BeautifulSoup(tr, 'html.parser')
print()

for row in table.find_all('tr'):
    print(row.attrs)


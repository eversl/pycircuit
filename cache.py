from PyCircuit import Vector, Or, Memory, Register, VFalse, VTrue, \
    FeedbackVector, If, Enum, ConstantVector, And


class doIf:
    def __init__(self, cond):
        self.cond = cond

    def do(self, val):
        assert len(self.cond) == 1
        if isinstance(val, tuple):
            then_, else_ = val
        else:
            then_ = val
            else_ = {}
        for r in then_:
            doReg(r, then_[r])
        for r in else_:
            doReg(r, else_[r])


class doElif:
    def __init__(self, cond):
        self.cond = cond


class doElse:
    pass


class doCase:
    def __init__(self, cond):
        self.cond = cond

    def do(self, val):
        print("case", str(val))
        cases = val
        for c in cases:
            print("  ", str(c), ":")
            for r in cases[c]:
                doReg(r, cases[c][r])


def do(clk, dict):
    for r in dict:
        doReg(r, dict[r])


def doReg(r, val):
    if isinstance(r, doIf):
        r.do(val)
    elif isinstance(r, doCase):
        r.do(val)
    else:
        print(r, val)


def simple_ram(clk, addr, data, isWrite, init=None):
    res = Memory(clk, addr, data, isWrite, init)
    return Register(clk, bits=len(res)).connect(res)


"""
// Cache Memory (4way 4word)               //
// i_  means input port                    //
// o_  means output port                   //
// _p_  means data exchange with processor //
// _m_  means data exchange with memory    //
// Replacement policy is LRU (8bit)        //


`default_nettype none

  module cache(clk,
               rst,
               i_p_addr,
               i_p_byte_en,
               i_p_writedata,
               i_p_read,
               i_p_write,
               o_p_readdata,
               o_p_readdata_valid,
               o_p_waitrequest,

               o_m_addr,
               o_m_byte_en,
               o_m_writedata,
               o_m_read,
               o_m_write,
               i_m_readdata,
               i_m_readdata_valid,
               i_m_waitrequest,

               cnt_r,
               cnt_w,
               cnt_hit_r,
               cnt_hit_w,
               cnt_wb_r,
               cnt_wb_w);

    parameter cache_entry = 14;
    input wire         clk, rst;
    input wire [24:0]  i_p_addr;
    input wire [3:0]   i_p_byte_en;
    input wire [31:0]  i_p_writedata;
    input wire         i_p_read, i_p_write;
    output reg [31:0]  o_p_readdata;
    output reg         o_p_readdata_valid;
    output wire        o_p_waitrequest;

    output reg [25:0]  o_m_addr;
    output wire [3:0]  o_m_byte_en;
    output reg [127:0] o_m_writedata;
    output reg         o_m_read, o_m_write;
    input wire [127:0] i_m_readdata;
    input wire         i_m_readdata_valid;
    input wire         i_m_waitrequest;

    output reg [31:0]  cnt_r;
    output reg [31:0]  cnt_w;
    output reg [31:0]  cnt_hit_r;
    output reg [31:0]  cnt_hit_w;
    output reg [31:0]  cnt_wb_r;
    output reg [31:0]  cnt_wb_w;

    wire [3:0]    hit;
    wire [3:0]    modify;
    wire [3:0]    miss;
    wire [3:0]    valid;
    wire [127:0]  readdata0, readdata1, readdata2, readdata3;
    wire [127:0]  writedata;
    wire          write0, write1, write2, write3;
    wire [3:0]    word_en;
    wire [3:0] 	  byte_en;
    wire [22:0]   addr;
    wire [22:0]   wb_addr0, wb_addr1, wb_addr2, wb_addr3;
    wire [7:0] 	  r_cm_data;
    wire [1:0] 	  hit_num;

    reg  [2:0] 	  state;
    reg  [127:0]  writedata_buf;
    reg  [24:0]   write_addr_buf;
    reg  [3:0] 	  byte_en_buf;
    reg 		  write_buf, read_buf;
    reg  [3:0]    write_set;
    reg  [3:0]    fetch_write;
    reg  [7:0] 	  w_cm_data;
    reg 		  w_cm;

    localparam IDLE = 0;
    localparam COMP = 1;
    localparam HIT  = 2;
    localparam FETCH1 = 3;
    localparam FETCH2 = 4;
    localparam FETCH3 = 5;
    localparam WB1 = 6;
    localparam WB2 = 7;


`ifdef SIM
    integer i;

    initial begin
        for(i = 0; i <=(2**cache_entry-1); i=i+1) begin
	        ram_hot.mem[i] = 0;
        end
    end
`endif

    simple_ram #(.width(8), .widthad(cache_entry)) ram_hot(clk, addr[cache_entry-1:0], w_cm, w_cm_data, addr[cache_entry-1:0], r_cm_data);

    set #(.cache_entry(cache_entry))
    set0(.clk(clk),
         .rst(rst),
         .entry(addr[cache_entry-1:0]),
         .o_tag(addr[22:cache_entry]),
         .writedata(writedata),
         .byte_en(byte_en),
         .write(write0),
         .word_en(word_en), // 4word r/w change
         .readdata(readdata0),
         .wb_addr(wb_addr0),
         .hit(hit[0]),
         .modify(modify[0]),
         .miss(miss[0]),
         .valid(valid[0]),
         .read_miss(read_buf));

    set #(.cache_entry(cache_entry))
    set1(.clk(clk),
         .rst(rst),
         .entry(addr[cache_entry-1:0]),
         .o_tag(addr[22:cache_entry]),
         .writedata(writedata),
         .byte_en(byte_en),
         .write(write1),
         .word_en(word_en), // 4word r/w change
         .readdata(readdata1),
         .wb_addr(wb_addr1),
         .hit(hit[1]),
         .modify(modify[1]),
         .miss(miss[1]),
         .valid(valid[1]),
         .read_miss(read_buf));

    set #(.cache_entry(cache_entry))
    set2(.clk(clk),
         .rst(rst),
         .entry(addr[cache_entry-1:0]),
         .o_tag(addr[22:cache_entry]),
         .writedata(writedata),
         .byte_en(byte_en),
         .write(write2),
         .word_en(word_en), // 4word r/w change
         .readdata(readdata2),
         .wb_addr(wb_addr2),
         .hit(hit[2]),
         .modify(modify[2]),
         .miss(miss[2]),
         .valid(valid[2]),
         .read_miss(read_buf));

    set #(.cache_entry(cache_entry))
    set3(.clk(clk),
         .rst(rst),
         .entry(addr[cache_entry-1:0]),
         .o_tag(addr[22:cache_entry]),
         .writedata(writedata),
         .byte_en(byte_en),
         .write(write3),
         .word_en(word_en), // 4word r/w change
         .readdata(readdata3),
         .wb_addr(wb_addr3),
         .hit(hit[3]),
         .modify(modify[3]),
         .miss(miss[3]),
         .valid(valid[3]),
         .read_miss(read_buf));

    assign writedata = (|fetch_write) ?	i_m_readdata : writedata_buf; //128bit
    assign write0 = (fetch_write[0]) ? i_m_readdata_valid : write_set[0];
    assign write1 = (fetch_write[1]) ? i_m_readdata_valid : write_set[1];
    assign write2 = (fetch_write[2]) ? i_m_readdata_valid : write_set[2];
    assign write3 = (fetch_write[3]) ? i_m_readdata_valid : write_set[3];
    assign addr = (o_p_waitrequest) ? write_addr_buf[24:2] : i_p_addr[24:2]; // set module input addr is 23bit
    assign byte_en = (|fetch_write) ? 4'b1111 : byte_en_buf;
    assign o_p_waitrequest = (state != IDLE);
    assign o_m_byte_en = 4'b1111;

    assign hit_num = (hit[0]) ? 0 : (hit[1]) ? 1 : (hit[2]) ? 2 : 3;
    assign word_en = (|fetch_write) ? 4'b1111 :
                     (write_addr_buf[1:0] == 2'b00) ? 4'b0001 :
                     (write_addr_buf[1:0] == 2'b01) ? 4'b0010 :
                     (write_addr_buf[1:0] == 2'b10) ? 4'b0100 : 4'b1000;

    always @(posedge clk) begin
        if(rst) begin
            o_p_readdata_valid <= 0;
            {o_m_read, o_m_write} <= 0;
            o_m_addr <= 0;
            write_addr_buf <= 0;
            byte_en_buf <= 0;
            writedata_buf <= 0;
            {write_buf, read_buf} <= 0;
            write_set <= 0;
            fetch_write <= 0;
            {cnt_r, cnt_w} <= 0;
            {cnt_hit_r, cnt_hit_w} <= 0;
            {cnt_wb_r, cnt_wb_w} <= 0;
            state <= IDLE;
        end
        else begin
            case (state)
                IDLE: begin
                    write_set <= 0;
                    o_p_readdata_valid <= 0;
                    writedata_buf <= {i_p_writedata, i_p_writedata, i_p_writedata, i_p_writedata};
                    write_addr_buf <= i_p_addr;
                    byte_en_buf <= i_p_byte_en;
                    write_buf <= i_p_write;
                    read_buf <= i_p_read;
                    if(i_p_read) begin
                        state <= COMP;
                        cnt_r <= cnt_r + 1;
                    end else if(i_p_write) begin
                        state <= COMP;
                        cnt_w <= cnt_w + 1;
                    end
                end
                COMP: begin
                    if((|hit) && write_buf) begin
                        state <= HIT;
                        write_set <= hit;
                        cnt_hit_w <= cnt_hit_w + 1;
                        w_cm_data <= (r_cm_data[1:0] == hit_num) ? {r_cm_data[1:0], r_cm_data[7:2]} :
                                     (r_cm_data[3:2] == hit_num) ? {r_cm_data[3:2], r_cm_data[7:4], r_cm_data[1:0]} :
                                     (r_cm_data[5:4] == hit_num) ? {r_cm_data[5:4], r_cm_data[7:6], r_cm_data[3:0]} : r_cm_data;
                        w_cm <= 1;
                    end else if((|hit) && read_buf) begin
                        case(write_addr_buf[1:0])
                            2'b00: o_p_readdata <= (hit[0]) ? readdata0[31:0] : (hit[1]) ? readdata1[31:0] : (hit[2]) ? readdata2[31:0] : readdata3[31:0];
                            2'b01: o_p_readdata <= (hit[0]) ? readdata0[63:32] : (hit[1]) ? readdata1[63:32] : (hit[2]) ? readdata2[63:32] : readdata3[63:32];
                            2'b10: o_p_readdata <= (hit[0]) ? readdata0[95:64] : (hit[1]) ? readdata1[95:64] : (hit[2]) ? readdata2[95:64] : readdata3[95:64];
                            2'b11: o_p_readdata <= (hit[0]) ? readdata0[127:96] : (hit[1]) ? readdata1[127:96] : (hit[2]) ? readdata2[127:96] : readdata3[127:96];
                        endcase
                        o_p_readdata_valid <= 1;
                        w_cm_data <= (r_cm_data[1:0] == hit_num) ? {r_cm_data[1:0], r_cm_data[7:2]} :
                                     (r_cm_data[3:2] == hit_num) ? {r_cm_data[3:2], r_cm_data[7:4], r_cm_data[1:0]} :
                                     (r_cm_data[5:4] == hit_num) ? {r_cm_data[5:4], r_cm_data[7:6], r_cm_data[3:0]} : r_cm_data;
                        w_cm <= 1;
                        cnt_hit_r <= cnt_hit_r + 1;
                        state <= IDLE;
                    end else if(!(&valid) || miss[r_cm_data[1:0]]) begin
                        state <= FETCH1;
                        if(!valid[0]) begin
                            fetch_write <= 4'b0001;
                            w_cm_data <= 8'b11100100;
                            w_cm <= 1;
                        end else if(!valid[1]) begin
                            fetch_write <= 4'b0010;
                            w_cm_data <= (r_cm_data[1:0] == 2'b01) ? {r_cm_data[1:0], r_cm_data[7:2]} :
                                         (r_cm_data[3:2] == 2'b01) ? {r_cm_data[3:2], r_cm_data[7:4], r_cm_data[1:0]} :
                                         (r_cm_data[5:4] == 2'b01) ? {r_cm_data[5:4], r_cm_data[7:6], r_cm_data[3:0]} : r_cm_data;
                            w_cm <= 1;
                        end else if(!valid[2]) begin
                            fetch_write <= 4'b0100;
                            w_cm_data <= (r_cm_data[1:0] == 2'b10) ? {r_cm_data[1:0], r_cm_data[7:2]} :
                                         (r_cm_data[3:2] == 2'b10) ? {r_cm_data[3:2], r_cm_data[7:4], r_cm_data[1:0]} :
                                         (r_cm_data[5:4] == 2'b10) ? {r_cm_data[5:4], r_cm_data[7:6], r_cm_data[3:0]} : r_cm_data;
                            w_cm <= 1;
                        end else if(!valid[3]) begin
                            fetch_write <= 4'b1000;
                            w_cm_data <= (r_cm_data[1:0] == 2'b11) ? {r_cm_data[1:0], r_cm_data[7:2]} :
                                         (r_cm_data[3:2] == 2'b11) ? {r_cm_data[3:2], r_cm_data[7:4], r_cm_data[1:0]} :
                                         (r_cm_data[5:4] == 2'b11) ? {r_cm_data[5:4], r_cm_data[7:6], r_cm_data[3:0]} : r_cm_data;
                            w_cm <= 1;
                        end else if(miss[r_cm_data[1:0]]) begin
                            if(r_cm_data[1:0] == 2'b00) fetch_write <= 4'b0001;
                            else if(r_cm_data[1:0] == 2'b01) fetch_write <= 4'b0010;
                            else if(r_cm_data[1:0] == 2'b10) fetch_write <= 4'b0100;
                            else if(r_cm_data[1:0] == 2'b11) fetch_write <= 4'b1000;
                            w_cm_data <= {r_cm_data[1:0], r_cm_data[7:2]};
                            w_cm <= 1;
                        end
                        o_m_addr <= {write_addr_buf[24:2], 3'b000};
                        o_m_read <= 1;
                    end else begin
                        state <= WB1;
                        if(r_cm_data[1:0] == 2'b00) fetch_write <= 4'b0001;
                        else if(r_cm_data[1:0] == 2'b01) fetch_write <= 4'b0010;
                        else if(r_cm_data[1:0] == 2'b10) fetch_write <= 4'b0100;
                        else if(r_cm_data[1:0] == 2'b11) fetch_write <= 4'b1000;
                        w_cm_data <= {r_cm_data[1:0], r_cm_data[7:2]};
                        w_cm <= 1;
                        if(read_buf) cnt_wb_r <= cnt_wb_r + 1;
                        else if(write_buf) cnt_wb_w <= cnt_wb_w + 1;
                    end
                end
                HIT: begin
                    w_cm <= 0;
                    write_set <= 0;
                    state <= IDLE;
                end //1/13
                FETCH1: begin
                    w_cm <= 0;
                    if(!i_m_waitrequest) begin
                        o_m_read <= 0;
                        state <= FETCH2;
                    end
                end
                FETCH2: begin
                    if(i_m_readdata_valid) begin
                        fetch_write <= 0;            //add 3/9
                        if(write_buf) begin
                            state <= FETCH3;
                            write_set <= fetch_write;
		                end else if(read_buf) begin
                            state <= IDLE;
		                    o_p_readdata_valid <= 1;
		                    case(write_addr_buf[1:0])
		                        2'b00: o_p_readdata <= i_m_readdata[ 31: 0];
		                        2'b01: o_p_readdata <= i_m_readdata[ 63:32];
		                        2'b10: o_p_readdata <= i_m_readdata[ 95:64];
		                        2'b11: o_p_readdata <= i_m_readdata[127:96];
		                    endcase
		                end
                    end
                end
                FETCH3: begin
                    state <= IDLE;
                    write_set <= 0;
                end
                WB1: begin
                    w_cm <= 0;
                    o_m_addr <= (fetch_write[0]) ? {wb_addr0, 3'b000} :
                                (fetch_write[1]) ? {wb_addr1, 3'b000} :
                                (fetch_write[2]) ? {wb_addr2, 3'b000} : {wb_addr3, 3'b000};
                    o_m_writedata <= (fetch_write[0]) ? readdata0 :
                                     (fetch_write[1]) ? readdata1 :
                                     (fetch_write[2]) ? readdata2 : readdata3;
                    o_m_write <= 1;
                    state <= WB2;
                end
                WB2: begin
                    if(!i_m_waitrequest) begin
                        o_m_write <= 0;
                        o_m_addr <= {write_addr_buf[24:2], 3'b000};
                        o_m_read <= 1;
                        state <= FETCH1;
                    end
                end
            endcase // case (state)
        end
    end

endmodule // cache
"""

#     localparam IDLE = 0;
#     localparam COMP = 1;
#     localparam HIT  = 2;
#     localparam FETCH1 = 3;
#     localparam FETCH2 = 4;
#     localparam FETCH3 = 5;
#     localparam WB1 = 6;
#     localparam WB2 = 7;

STATES = Enum('IDLE', 'COMP', 'HIT', 'FETCH1', 'FETCH2', 'FETCH3', 'WB1', 'WB2')


def Cache(clk,
          rst,
          i_p_addr,
          i_p_byte_en,
          i_p_writedata,
          i_p_read,
          i_p_write,

          i_m_readdata,
          i_m_readdata_valid,
          i_m_waitrequest
          ):
    #     parameter cache_entry = 14;
    #     input wire         clk, rst;
    #     input wire [24:0]  i_p_addr;
    #     input wire [3:0]   i_p_byte_en;
    #     input wire [31:0]  i_p_writedata;
    #     input wire         i_p_read, i_p_write;
    #     output reg [31:0]  o_p_readdata;
    #     output reg         o_p_readdata_valid;
    #     output wire        o_p_waitrequest;
    #
    #     output reg [25:0]  o_m_addr;
    #     output wire [3:0]  o_m_byte_en;
    #     output reg [127:0] o_m_writedata;
    #     output reg         o_m_read, o_m_write;
    #     input wire [127:0] i_m_readdata;
    #     input wire         i_m_readdata_valid;
    #     input wire         i_m_waitrequest;
    #
    #     output reg [31:0]  cnt_r;
    #     output reg [31:0]  cnt_w;
    #     output reg [31:0]  cnt_hit_r;
    #     output reg [31:0]  cnt_hit_w;
    #     output reg [31:0]  cnt_wb_r;
    #     output reg [31:0]  cnt_wb_w;

    # counters
    cnt_r = Register(clk, 0, 32, name="cnt_r")
    cnt_w = Register(clk, 0, 32, name="cnt_w")
    cnt_hit_r = Register(clk, 0, 32, name="cnt_hit_r")
    cnt_hit_w = Register(clk, 0, 32, name="cnt_hit_w")
    cnt_wb_r = Register(clk, 0, 32, name="cnt_wb_r")
    cnt_wb_w = Register(clk, 0, 32, name="cnt_wb_w")

    cache_entry = 14

    hit = FeedbackVector(0, 4, name="hit")
    modify = FeedbackVector(0, 4)
    miss = FeedbackVector(0, 4)
    valid = FeedbackVector(0, 4)
    readdata0 = FeedbackVector(0, 128)
    readdata1 = FeedbackVector(0, 128)
    readdata2 = FeedbackVector(0, 128)
    readdata3 = FeedbackVector(0, 128)
    word_en = FeedbackVector(0, 4)
    wb_addr0 = FeedbackVector(0, 22)
    wb_addr1 = FeedbackVector(0, 22)
    wb_addr2 = FeedbackVector(0, 22)
    wb_addr3 = FeedbackVector(0, 22)
    r_cm_data = FeedbackVector(0, 8)
    hit_num = FeedbackVector(0, 2)

    o_p_readdata = Register(clk, 0, 32)
    o_p_readdata_valid = Register(clk, name="o_p_readdata_valid")
    o_m_addr = Register(clk, 0, 26, name="o_m_addr")
    o_m_writedata = Register(clk, 0, 128, name="o_m_writedata")
    o_m_read = Register(clk, name="o_m_read")
    o_m_write = Register(clk, name="o_m_write")

    state = Register(clk, STATES['IDLE'], name="state")

    writedata_buf = Register(clk, 0, 128, name="writedata_buf")
    write_addr_buf = Register(clk, 0, 25, name="write_addr_buf")
    byte_en_buf = Register(clk, 0, 4, name="byte_en_buf")
    write_buf = Register(clk, 0, 1, name="write_buf")
    read_buf = Register(clk, 0, 1, name="read_buf")
    write_set = Register(clk, 0, 4, name="write_set")
    fetch_write = Register(clk, 0, 4, name="fetch_write")
    w_cm_data = Register(clk, 0, 8, name="w_cm_data")
    w_cm = Register(clk, 0, 1, name="w_cm")

    writedata = If(fetch_write.reduce(Or), i_m_readdata, writedata_buf)  # 128bit
    write0 = If(fetch_write[0], i_m_readdata_valid, write_set[0])
    write1 = If(fetch_write[1], i_m_readdata_valid, write_set[1])
    write2 = If(fetch_write[2], i_m_readdata_valid, write_set[2])
    write3 = If(fetch_write[3], i_m_readdata_valid, write_set[3])
    o_p_waitrequest = (state != STATES['IDLE'])
    addr = If(o_p_waitrequest, write_addr_buf[2:25], i_p_addr[2:25])  # set module input addr is 23bit
    byte_en = If(fetch_write.reduce(Or), ConstantVector(-1, 4), byte_en_buf)
    o_m_byte_en = ConstantVector(-1, 4)

    hit_num = If(hit[0], ConstantVector(0, 2),
                 If(hit[1], ConstantVector(1, 2), If(hit[2], ConstantVector(2, 2), ConstantVector(3, 2))))
    word_en.connect(If(fetch_write.reduce(Or), ConstantVector(-1, 4),
                       If(write_addr_buf[0:2] == ConstantVector(0, 2), ConstantVector(1, 4),
                          If(write_addr_buf[0:2] == ConstantVector(1, 2), ConstantVector(2, 4),
                             If(write_addr_buf[0:2] == ConstantVector(2, 2), ConstantVector(4, 4),
                                ConstantVector(8, 4))))))

    # simple_ram #(.width(8), .widthad(cache_entry)) ram_hot(clk, addr[cache_entry-1:0], w_cm, w_cm_data, addr[cache_entry-1:0], r_cm_data);
    # ram_hot
    entry = addr[0:cache_entry]
    tag = addr[cache_entry:23]
    r_cm_data = simple_ram(clk, entry, w_cm_data, w_cm)

    # set #(.cache_entry(cache_entry))
    # set0(.clk(clk),
    #      .rst(rst),
    #      .entry(addr[cache_entry-1:0]),
    #      .o_tag(addr[22:cache_entry]),
    #      .writedata(writedata),
    #      .byte_en(byte_en),
    #      .write(write0),
    #      .word_en(word_en), // 4word r/w change
    #      .readdata(readdata0),
    #      .wb_addr(wb_addr0),
    #      .hit(hit[0]),
    #      .modify(modify[0]),
    #      .miss(miss[0]),
    #      .valid(valid[0]),
    #      .read_miss(read_buf));
    set0 = Set(clk, entry=entry, o_tag=tag, writedata=writedata, byte_en=byte_en,
               write=write0, word_en=word_en, read_miss=read_buf)
    #
    # set #(.cache_entry(cache_entry))
    # set1(.clk(clk),
    #      .rst(rst),
    #      .entry(addr[cache_entry-1:0]),
    #      .o_tag(addr[22:cache_entry]),
    #      .writedata(writedata),
    #      .byte_en(byte_en),
    #      .write(write1),
    #      .word_en(word_en), // 4word r/w change
    #      .readdata(readdata1),
    #      .wb_addr(wb_addr1),
    #      .hit(hit[1]),
    #      .modify(modify[1]),
    #      .miss(miss[1]),
    #      .valid(valid[1]),
    #      .read_miss(read_buf));
    set1 = Set(clk, entry=entry, o_tag=tag, writedata=writedata, byte_en=byte_en,
               write=write1, word_en=word_en, read_miss=read_buf)
    #
    # set #(.cache_entry(cache_entry))
    # set2(.clk(clk),
    #      .rst(rst),
    #      .entry(addr[cache_entry-1:0]),
    #      .o_tag(addr[22:cache_entry]),
    #      .writedata(writedata),
    #      .byte_en(byte_en),
    #      .write(write2),
    #      .word_en(word_en), // 4word r/w change
    #      .readdata(readdata2),
    #      .wb_addr(wb_addr2),
    #      .hit(hit[2]),
    #      .modify(modify[2]),
    #      .miss(miss[2]),
    #      .valid(valid[2]),
    #      .read_miss(read_buf));
    set2 = Set(clk, entry=entry, o_tag=tag, writedata=writedata, byte_en=byte_en,
               write=write2, word_en=word_en, read_miss=read_buf)
    #
    # set #(.cache_entry(cache_entry))
    # set3(.clk(clk),
    #      .rst(rst),
    #      .entry(addr[cache_entry-1:0]),
    #      .o_tag(addr[22:cache_entry]),
    #      .writedata(writedata),
    #      .byte_en(byte_en),
    #      .write(write3),
    #      .word_en(word_en), // 4word r/w change
    #      .readdata(readdata3),
    #      .wb_addr(wb_addr3),
    #      .hit(hit[3]),
    #      .modify(modify[3]),
    #      .miss(miss[3]),
    #      .valid(valid[3]),
    #      .read_miss(read_buf));
    set3 = Set(clk, entry=entry, o_tag=tag, writedata=writedata, byte_en=byte_en,
               write=write3, word_en=word_en, read_miss=read_buf)
    readdata0_, wb_addr0_, hit0, modify0, miss0, valid0 = set0
    readdata0.connect(readdata0_)
    wb_addr0.connect(wb_addr0_)
    readdata1_, wb_addr1_, hit1, modify1, miss1, valid1 = set1
    readdata1.connect(readdata1_)
    wb_addr1.connect(wb_addr1_)
    readdata2_, wb_addr2_, hit2, modify2, miss2, valid2 = set2
    readdata2.connect(readdata2_)
    wb_addr2.connect(wb_addr2_)
    readdata3_, wb_addr3_, hit3, modify3, miss3, valid3 = set3
    readdata3.connect(readdata3_)
    wb_addr3.connect(wb_addr3_)

    hits = Vector.concatRev(hit0, hit1, hit2, hit3)
    hit.connect(hits)
    modify.connect(Vector.concatRev(modify0, modify1, modify2, modify3))
    miss.connect(Vector.concatRev(miss0, miss1, miss2, miss3))
    valid.connect(Vector.concatRev(valid0, valid1, valid2, valid3))

    do(clk, {

        doIf(rst): {
            o_p_readdata_valid: 0,
            o_m_read: 0, o_m_write: 0,
            o_m_addr: 0,
            write_addr_buf: 0,
            byte_en_buf: 0,
            writedata_buf: 0,
            write_buf: 0, read_buf: 0,
            write_set: 0,
            fetch_write: 0,
            cnt_r: 0, cnt_w: 0,
            cnt_hit_r: 0, cnt_hit_w: 0,
            cnt_wb_r: 0, cnt_wb_w: 0,
            state: "IDLE"
        }, doElse(): {
            doCase(state): {
                "IDLE": {
                    write_set: 0,
                    o_p_readdata_valid: 0,
                    writedata_buf: {i_p_writedata, i_p_writedata, i_p_writedata, i_p_writedata},
                    write_addr_buf: i_p_addr,
                    byte_en_buf: i_p_byte_en,
                    write_buf: i_p_write,
                    read_buf: i_p_read,
                    doIf(i_p_read):
                        {
                            state: "COMP",
                            cnt_r: cnt_r + 1
                        }, doElse(): {
                        doIf(i_p_write): {
                            state: "COMP",
                            cnt_w: cnt_w + 1
                        }
                    }
                }, "COMP": {
                    doIf(hit.reduce(Or) & write_buf): {
                        state: "HIT",
                        write_set: hit,
                        cnt_hit_w: cnt_hit_w + 1,
                        w_cm_data: If(r_cm_data[0:2] == hit_num, Vector.concatRev(r_cm_data[0:2], r_cm_data[2:8]),
                                      If(r_cm_data[2:4] == hit_num,
                                         Vector.concatRev(r_cm_data[2:4], r_cm_data[4:8], r_cm_data[0:2]),
                                         If(r_cm_data[4:6] == hit_num,
                                            Vector.concatRev(r_cm_data[4:6], r_cm_data[6:8], r_cm_data[0:4]),
                                            r_cm_data))),
                        w_cm: 1
                    },
                    doElif(hit.reduce(Or) & read_buf): {
                        doCase(write_addr_buf[0:2]): {
                            0b00: {o_p_readdata: If(hit[0], readdata0[0:32], If(hit[1], readdata1[0:32],
                                                                                If(hit[2], readdata2[0:32],
                                                                                   readdata3[0:32])))},
                            0b01: {o_p_readdata: If(hit[0], readdata0[32:64], If(hit[1], readdata1[32:64],
                                                                                 If(hit[2], readdata2[32:64],
                                                                                    readdata3[32:64])))},
                            0b10: {o_p_readdata: If(hit[0], readdata0[64:96], If(hit[1], readdata1[64:96],
                                                                                 If(hit[2], readdata2[64:96],
                                                                                    readdata3[64:96])))},
                            0b11: {o_p_readdata: If(hit[0], readdata0[96:128], If(hit[1], readdata1[96:128],
                                                                                  If(hit[2], readdata2[96:128],
                                                                                     readdata3[96:128])))}
                        },
                        o_p_readdata_valid: 1,
                        w_cm_data: If(r_cm_data[0:2] == hit_num, Vector.concatRev(r_cm_data[0:2], r_cm_data[2:8]),
                                      If(r_cm_data[2:4] == hit_num,
                                         Vector.concatRev(r_cm_data[2:4], r_cm_data[4:8], r_cm_data[0:2]),
                                         If(r_cm_data[4:6] == hit_num,
                                            Vector.concatRev(r_cm_data[4:6], r_cm_data[6:8], r_cm_data[0:4]),
                                            r_cm_data))),
                        w_cm: 1,
                        cnt_hit_r: cnt_hit_r + 1,
                        state: "IDLE"
                    }, doElif(~valid.reduce(And) | miss[r_cm_data[0:2]]): {
                        state: "FETCH1",
                        doIf(~valid[0]): {
                            fetch_write: 0b0001,
                            w_cm_data: 0b11100100,
                            w_cm: 1,
                        }, doElif(~valid[1]): {
                            fetch_write: 0b0010,
                            w_cm_data: If(r_cm_data[0:2] == 0b01, Vector.concatRev(r_cm_data[0:2], r_cm_data[2:8]),
                                          If(r_cm_data[2:4] == 0b01,
                                             Vector.concatRev(r_cm_data[2:4], r_cm_data[4:8], r_cm_data[0:2]),
                                             If(r_cm_data[4:6] == 0b01,
                                                Vector.concatRev(r_cm_data[4:6], r_cm_data[6:8], r_cm_data[0:4]),
                                                r_cm_data))),
                            w_cm: 1
                        }, doElif(~valid[2]): {
                            fetch_write: 0b0100,
                            w_cm_data: If(r_cm_data[0:2] == 0b10, Vector.concatRev(r_cm_data[0:2], r_cm_data[2:8]),
                                          If(r_cm_data[2:4] == 0b10,
                                             Vector.concatRev(r_cm_data[2:4], r_cm_data[4:8], r_cm_data[0:2]),
                                             If(r_cm_data[4:6] == 0b10,
                                                Vector.concatRev(r_cm_data[4:6], r_cm_data[6:8], r_cm_data[0:4]),
                                                r_cm_data))),
                            w_cm: 1
                        }, doElif(~valid[3]): {
                            fetch_write: 0b1000,
                            w_cm_data: If(r_cm_data[0:2] == 0b11, Vector.concatRev(r_cm_data[0:2], r_cm_data[2:8]),
                                          If(r_cm_data[2:4] == 0b11,
                                             Vector.concatRev(r_cm_data[2:4], r_cm_data[4:8], r_cm_data[0:2]),
                                             If(r_cm_data[4:6] == 0b11,
                                                Vector.concatRev(r_cm_data[4:6], r_cm_data[6:8], r_cm_data[0:4]),
                                                r_cm_data))),
                            w_cm: 1
                        }, doElif(miss[r_cm_data[0:2]]): {
                            doIf(r_cm_data[0:2] == 0b00): {fetch_write: 0b0001},
                            doElif(r_cm_data[0:2] == 0b01): {fetch_write: 0b0010},
                            doElif(r_cm_data[0:2] == 0b10): {fetch_write: 0b0100},
                            doElif(r_cm_data[0:2] == 0b11): {fetch_write: 0b1000},
                            doElse(): {
                                w_cm_data: Vector.concatRev(r_cm_data[0:2], r_cm_data[2:8]),
                                w_cm: 1
                            }
                        },
                        o_m_addr: write_addr_buf[2:25].extendTo(len(o_m_addr), LSB=True),
                        o_m_read: 1
                    }, doElse(): {
                        state: "WB1",
                        doIf(r_cm_data[0:2] == 0b00): {fetch_write: 0b0001},
                        doElif(r_cm_data[0:2] == 0b01): {fetch_write: 0b0010},
                        doElif(r_cm_data[0:2] == 0b10): {fetch_write: 0b0100},
                        doElif(r_cm_data[0:2] == 0b11): {fetch_write: 0b1000},
                        w_cm_data: Vector.concatRev(r_cm_data[0:2], r_cm_data[2:8]),
                        w_cm: 1,
                        doIf(read_buf): {cnt_wb_r: cnt_wb_r + 1},
                        doElif(write_buf): {cnt_wb_w: cnt_wb_w + 1},
                    }
                }, "HIT": {
                    w_cm: 0,
                    write_set: 0,
                    state: "IDLE",
                }, "FETCH1": {
                    w_cm: 0,
                    doIf(~i_m_waitrequest): {
                        o_m_read: 0,
                        state: "FETCH2",
                    }
                }, "FETCH2": {
                    doIf(i_m_readdata_valid): {
                        fetch_write: 0,
                        doIf(write_buf): {
                            state: "FETCH3",
                            write_set: fetch_write,
                        }, doElif(read_buf): {
                            state: "IDLE",
                            o_p_readdata_valid: 1,
                            doCase(write_addr_buf[0:2]): {
                                0b00: {o_p_readdata: i_m_readdata[0:32]},
                                0b01: {o_p_readdata: i_m_readdata[32:64]},
                                0b10: {o_p_readdata: i_m_readdata[64:96]},
                                0b11: {o_p_readdata: i_m_readdata[96:128]}
                            }
                        }
                    }
                }, "FETCH3": {
                    state: "IDLE",
                    write_set: 0
                }, "WB1": {
                    w_cm: 0,
                    o_m_addr: If(fetch_write[0], wb_addr0.extendTo(len(o_m_addr), LSB=True),
                                 If(fetch_write[1], wb_addr1.extendTo(len(o_m_addr), LSB=True),
                                    If(fetch_write[2], wb_addr2.extendTo(len(o_m_addr), LSB=True),
                                       wb_addr3.extendTo(len(o_m_addr), LSB=True)))),
                    o_m_writedata: If(fetch_write[0], readdata0,
                                      If(fetch_write[1], readdata1,
                                         If(fetch_write[2], readdata2, readdata3))),
                    o_m_write: 1,
                    state: "WB2"
                }, "WB2": {
                    doIf(~i_m_waitrequest): {
                        o_m_write: 0,
                        o_m_addr: write_addr_buf[2:25].extendTo(len(o_m_addr), LSB=True),
                        o_m_read: 1,
                        state: "FETCH1"
                    }
                }
            }
        }
    })

    #
    # always @(posedge clk) begin
    #     if(rst) begin
    #         o_p_readdata_valid <= 0;
    #         {o_m_read, o_m_write} <= 0;
    #         o_m_addr <= 0;
    #         write_addr_buf <= 0;
    #         byte_en_buf <= 0;
    #         writedata_buf <= 0;
    #         {write_buf, read_buf} <= 0;
    #         write_set <= 0;
    #         fetch_write <= 0;
    #         {cnt_r, cnt_w} <= 0;
    #         {cnt_hit_r, cnt_hit_w} <= 0;
    #         {cnt_wb_r, cnt_wb_w} <= 0;
    #         state <= IDLE;
    #     end
    #     else begin
    #         case (state)
    #             IDLE: begin
    #                 write_set <= 0;
    #                 o_p_readdata_valid <= 0;
    #                 writedata_buf <= {i_p_writedata, i_p_writedata, i_p_writedata, i_p_writedata};
    #                 write_addr_buf <= i_p_addr;
    #                 byte_en_buf <= i_p_byte_en;
    #                 write_buf <= i_p_write;
    #                 read_buf <= i_p_read;
    #                 if(i_p_read) begin
    #                     state <= COMP;
    #                     cnt_r <= cnt_r + 1;
    #                 end else if(i_p_write) begin
    #                     state <= COMP;
    #                     cnt_w <= cnt_w + 1;
    #                 end
    #             end
    #             COMP: begin
    #                 if((|hit) && write_buf) begin
    #                     state <= HIT;
    #                     write_set <= hit;
    #                     cnt_hit_w <= cnt_hit_w + 1;
    #                     w_cm_data <= (r_cm_data[1:0] == hit_num) ? {r_cm_data[1:0], r_cm_data[7:2]} :
    #                                  (r_cm_data[3:2] == hit_num) ? {r_cm_data[3:2], r_cm_data[7:4], r_cm_data[1:0]} :
    #                                  (r_cm_data[5:4] == hit_num) ? {r_cm_data[5:4], r_cm_data[7:6], r_cm_data[3:0]} : r_cm_data;
    #                     w_cm <= 1;
    #                 end else if((|hit) && read_buf) begin
    #                     case(write_addr_buf[1:0])
    #                         2'b00: o_p_readdata <= (hit[0]) ? readdata0[31:0] : (hit[1]) ? readdata1[31:0] : (hit[2]) ? readdata2[31:0] : readdata3[31:0];
    #                         2'b01: o_p_readdata <= (hit[0]) ? readdata0[63:32] : (hit[1]) ? readdata1[63:32] : (hit[2]) ? readdata2[63:32] : readdata3[63:32];
    #                         2'b10: o_p_readdata <= (hit[0]) ? readdata0[95:64] : (hit[1]) ? readdata1[95:64] : (hit[2]) ? readdata2[95:64] : readdata3[95:64];
    #                         2'b11: o_p_readdata <= (hit[0]) ? readdata0[127:96] : (hit[1]) ? readdata1[127:96] : (hit[2]) ? readdata2[127:96] : readdata3[127:96];
    #                     endcase
    #                     o_p_readdata_valid <= 1;
    #                     w_cm_data <= (r_cm_data[1:0] == hit_num) ? {r_cm_data[1:0], r_cm_data[7:2]} :
    #                                  (r_cm_data[3:2] == hit_num) ? {r_cm_data[3:2], r_cm_data[7:4], r_cm_data[1:0]} :
    #                                  (r_cm_data[5:4] == hit_num) ? {r_cm_data[5:4], r_cm_data[7:6], r_cm_data[3:0]} : r_cm_data;
    #                     w_cm <= 1;
    #                     cnt_hit_r <= cnt_hit_r + 1;
    #                     state <= IDLE;
    #                 end else if(!(&valid) || miss[r_cm_data[1:0]]) begin
    #                     state <= FETCH1;
    #                     if(!valid[0]) begin
    #                         fetch_write <= 4'b0001;
    #                         w_cm_data <= 8'b11100100;
    #                         w_cm <= 1;
    #                     end else if(!valid[1]) begin
    #                         fetch_write <= 4'b0010;
    #                         w_cm_data <= (r_cm_data[1:0] == 2'b01) ? {r_cm_data[1:0], r_cm_data[7:2]} :
    #                                      (r_cm_data[3:2] == 2'b01) ? {r_cm_data[3:2], r_cm_data[7:4], r_cm_data[1:0]} :
    #                                      (r_cm_data[5:4] == 2'b01) ? {r_cm_data[5:4], r_cm_data[7:6], r_cm_data[3:0]} : r_cm_data;
    #                         w_cm <= 1;
    #                     end else if(!valid[2]) begin
    #                         fetch_write <= 4'b0100;
    #                         w_cm_data <= (r_cm_data[1:0] == 2'b10) ? {r_cm_data[1:0], r_cm_data[7:2]} :
    #                                      (r_cm_data[3:2] == 2'b10) ? {r_cm_data[3:2], r_cm_data[7:4], r_cm_data[1:0]} :
    #                                      (r_cm_data[5:4] == 2'b10) ? {r_cm_data[5:4], r_cm_data[7:6], r_cm_data[3:0]} : r_cm_data;
    #                         w_cm <= 1;
    #                     end else if(!valid[3]) begin
    #                         fetch_write <= 4'b1000;
    #                         w_cm_data <= (r_cm_data[1:0] == 2'b11) ? {r_cm_data[1:0], r_cm_data[7:2]} :
    #                                      (r_cm_data[3:2] == 2'b11) ? {r_cm_data[3:2], r_cm_data[7:4], r_cm_data[1:0]} :
    #                                      (r_cm_data[5:4] == 2'b11) ? {r_cm_data[5:4], r_cm_data[7:6], r_cm_data[3:0]} : r_cm_data;
    #                         w_cm <= 1;
    #                     end else if(miss[r_cm_data[1:0]]) begin
    #                         if(r_cm_data[1:0] == 2'b00) fetch_write <= 4'b0001;
    #                         else if(r_cm_data[1:0] == 2'b01) fetch_write <= 4'b0010;
    #                         else if(r_cm_data[1:0] == 2'b10) fetch_write <= 4'b0100;
    #                         else if(r_cm_data[1:0] == 2'b11) fetch_write <= 4'b1000;
    #                         w_cm_data <= {r_cm_data[1:0], r_cm_data[7:2]};
    #                         w_cm <= 1;
    #                     end
    #                     o_m_addr <= {write_addr_buf[24:2], 3'b000};
    #                     o_m_read <= 1;
    #                 end else begin
    #                     state <= WB1;
    #                     if(r_cm_data[1:0] == 2'b00) fetch_write <= 4'b0001;
    #                     else if(r_cm_data[1:0] == 2'b01) fetch_write <= 4'b0010;
    #                     else if(r_cm_data[1:0] == 2'b10) fetch_write <= 4'b0100;
    #                     else if(r_cm_data[1:0] == 2'b11) fetch_write <= 4'b1000;
    #                     w_cm_data <= {r_cm_data[1:0], r_cm_data[7:2]};
    #                     w_cm <= 1;
    #                     if(read_buf) cnt_wb_r <= cnt_wb_r + 1;
    #                     else if(write_buf) cnt_wb_w <= cnt_wb_w + 1;
    #                 end
    #             end
    #             HIT: begin
    #                 w_cm <= 0;
    #                 write_set <= 0;
    #                 state <= IDLE;
    #             end //1/13
    #             FETCH1: begin
    #                 w_cm <= 0;
    #                 if(!i_m_waitrequest) begin
    #                     o_m_read <= 0;
    #                     state <= FETCH2;
    #                 end
    #             end
    #             FETCH2: begin
    #                 if(i_m_readdata_valid) begin
    #                     fetch_write <= 0;            //add 3/9
    #                     if(write_buf) begin
    #                         state <= FETCH3;
    #                         write_set <= fetch_write;
    #                 end else if(read_buf) begin
    #                         state <= IDLE;
    #                     o_p_readdata_valid <= 1;
    #                     case(write_addr_buf[1:0])
    #                         2'b00: o_p_readdata <= i_m_readdata[ 31: 0];
    #                         2'b01: o_p_readdata <= i_m_readdata[ 63:32];
    #                         2'b10: o_p_readdata <= i_m_readdata[ 95:64];
    #                         2'b11: o_p_readdata <= i_m_readdata[127:96];
    #                     endcase
    #                 end
    #                 end
    #             end
    #             FETCH3: begin
    #                 state <= IDLE;
    #                 write_set <= 0;
    #             end
    #             WB1: begin
    #                 w_cm <= 0;
    #                 o_m_addr <= (fetch_write[0]) ? {wb_addr0, 3'b000} :
    #                             (fetch_write[1]) ? {wb_addr1, 3'b000} :
    #                             (fetch_write[2]) ? {wb_addr2, 3'b000} : {wb_addr3, 3'b000};
    #                 o_m_writedata <= (fetch_write[0]) ? readdata0 :
    #                                  (fetch_write[1]) ? readdata1 :
    #                                  (fetch_write[2]) ? readdata2 : readdata3;
    #                 o_m_write <= 1;
    #                 state <= WB2;
    #             end
    #             WB2: begin
    #                 if(!i_m_waitrequest) begin
    #                     o_m_write <= 0;
    #                     o_m_addr <= {write_addr_buf[24:2], 3'b000};
    #                     o_m_read <= 1;
    #                     state <= FETCH1;
    #                 end
    #             end
    #         endcase // case (state)
    #     end
    # end
    #
    return o_p_readdata, o_p_readdata_valid, o_p_waitrequest, o_m_addr, o_m_byte_en, o_m_writedata, o_m_read, o_m_write


"""
module set(clk,
           rst,
           entry,
           o_tag,
           writedata,
           byte_en,
           write,
           word_en,

           readdata,
           wb_addr,
           hit,
           modify,
           miss,
           valid,
           read_miss);

    parameter cache_entry = 14;

    input wire                    clk, rst;
    input wire [cache_entry-1:0]  entry;
    input wire [22-cache_entry:0] o_tag;
    input wire [127:0] 		      writedata;
    input wire [3:0] 		      byte_en;
    input wire       	          write;
    input wire [3:0]              word_en;
    input wire 			          read_miss;

    output wire [127:0] 		  readdata;
    output wire [22:0] 		      wb_addr;
    output wire 			      hit, modify, miss, valid;



    wire [22-cache_entry:0] 	 i_tag;
    wire 			             dirty;
    wire [24-cache_entry:0] 	 write_tag_data;

    assign hit = valid && (o_tag == i_tag);
    assign modify = valid && (o_tag != i_tag) && dirty;
    assign miss = !valid || ((o_tag != i_tag) && !dirty);

    assign wb_addr = {i_tag, entry};

    //write -> [3:0] write, writedata/readdata 32bit -> 128bit
    simple_ram #(.width(8), .widthad(cache_entry)) ram11_3(clk, entry, write && word_en[3]  && byte_en[3], writedata[127:120], entry, readdata[127:120]);
    simple_ram #(.width(8), .widthad(cache_entry)) ram11_2(clk, entry, write && word_en[3]  && byte_en[2], writedata[119:112], entry, readdata[119:112]);
    simple_ram #(.width(8), .widthad(cache_entry)) ram11_1(clk, entry, write && word_en[3]  && byte_en[1], writedata[111:104], entry, readdata[111:104]);
    simple_ram #(.width(8), .widthad(cache_entry)) ram11_0(clk, entry, write && word_en[3]  && byte_en[0], writedata[103:96], entry, readdata[103:96]);

    simple_ram #(.width(8), .widthad(cache_entry)) ram10_3(clk, entry, write && word_en[2]  && byte_en[3], writedata[95:88], entry, readdata[95:88]);
    simple_ram #(.width(8), .widthad(cache_entry)) ram10_2(clk, entry, write && word_en[2]  && byte_en[2], writedata[87:80], entry, readdata[87:80]);
    simple_ram #(.width(8), .widthad(cache_entry)) ram10_1(clk, entry, write && word_en[2]  && byte_en[1], writedata[79:72], entry, readdata[79:72]);
    simple_ram #(.width(8), .widthad(cache_entry)) ram10_0(clk, entry, write && word_en[2]  && byte_en[0], writedata[71:64], entry, readdata[71:64]);

    simple_ram #(.width(8), .widthad(cache_entry)) ram01_3(clk, entry, write && word_en[1]  && byte_en[3], writedata[63:56], entry, readdata[63:56]);
    simple_ram #(.width(8), .widthad(cache_entry)) ram01_2(clk, entry, write && word_en[1]  && byte_en[2], writedata[55:48], entry, readdata[55:48]);
    simple_ram #(.width(8), .widthad(cache_entry)) ram01_1(clk, entry, write && word_en[1]  && byte_en[1], writedata[47:40], entry, readdata[47:40]);
    simple_ram #(.width(8), .widthad(cache_entry)) ram01_0(clk, entry, write && word_en[1]  && byte_en[0], writedata[39:32], entry, readdata[39:32]);

    simple_ram #(.width(8), .widthad(cache_entry)) ram00_3(clk, entry, write && word_en[0]  && byte_en[3], writedata[31:24], entry, readdata[31:24]);
    simple_ram #(.width(8), .widthad(cache_entry)) ram00_2(clk, entry, write && word_en[0]  && byte_en[2], writedata[23:16], entry, readdata[23:16]);
    simple_ram #(.width(8), .widthad(cache_entry)) ram00_1(clk, entry, write && word_en[0]  && byte_en[1], writedata[15: 8], entry, readdata[15:8]);
    simple_ram #(.width(8), .widthad(cache_entry)) ram00_0(clk, entry, write && word_en[0]  && byte_en[0], writedata[ 7: 0], entry, readdata[ 7:0]);


    assign write_tag_data = (read_miss) ? {1'b0, 1'b1, o_tag} : (modify || miss ) ? {1'b1, 1'b1, o_tag} : {1'b1, 1'b1, i_tag};
    simple_ram #(.width(25-cache_entry), .widthad(cache_entry)) ram_tag(clk, entry, write, write_tag_data, entry, {dirty, valid, i_tag});

`ifdef SIM
    integer i;

    initial begin
        for(i = 0; i <=(2**cache_entry-1); i=i+1) begin
	        ram_tag.mem[i] = 0;
        end
    end
`endif

endmodule
"""


def Set(clk,
        entry,
        o_tag,
        writedata,
        byte_en,
        write,
        word_en,
        read_miss):
    cache_entry = 14
    #
    #     input wire                    clk, rst;
    #     input wire [cache_entry-1:0]  entry;
    #     input wire [22-cache_entry:0] o_tag;
    #     input wire [127:0] 		      writedata;
    #     input wire [3:0] 		      byte_en;
    #     input wire       	          write;
    #     input wire [3:0]              word_en;
    #     input wire 			          read_miss;
    #
    #     output wire [127:0] 		  readdata;
    #     output wire [22:0] 		      wb_addr;
    #     output wire 			      hit, modify, miss, valid;
    #
    #
    #
    #     wire [22-cache_entry:0] 	 i_tag;
    i_tag = FeedbackVector(0, 23 - cache_entry)
    #     wire 			             dirty;
    dirty = FeedbackVector(0)
    #     wire [24-cache_entry:0] 	 write_tag_data;

    valid = FeedbackVector(0)

    #
    hit = valid & (o_tag == i_tag)
    modify = valid & (o_tag != i_tag) & dirty
    miss = ~ valid & ((o_tag != i_tag) & ~ dirty)
    #
    wb_addr = Vector.concatRev(i_tag, entry)

    #     //write -> [3:0] write, writedata/readdata 32bit -> 128bit
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram11_3(clk, entry, write && word_en[3]  && byte_en[3], writedata[127:120], entry, readdata[127:120]);
    write_word_en_3 = write & word_en[3]
    ram11_3 = simple_ram(clk, entry, writedata[120:127], write_word_en_3 & byte_en[3])
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram11_2(clk, entry, write && word_en[3]  && byte_en[2], writedata[119:112], entry, readdata[119:112]);
    ram11_2 = simple_ram(clk, entry, writedata[112:120], write_word_en_3 & byte_en[2])
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram11_1(clk, entry, write && word_en[3]  && byte_en[1], writedata[111:104], entry, readdata[111:104]);
    ram11_1 = simple_ram(clk, entry, writedata[104:112], write_word_en_3 & byte_en[1])
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram11_0(clk, entry, write && word_en[3]  && byte_en[0], writedata[103:96], entry, readdata[103:96]);
    ram11_0 = simple_ram(clk, entry, writedata[96:104], write_word_en_3 & byte_en[0])
    #
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram10_3(clk, entry, write && word_en[2]  && byte_en[3], writedata[95:88], entry, readdata[95:88]);
    write_word_en_2 = write & word_en[2]
    ram10_3 = simple_ram(clk, entry, writedata[88:96], write_word_en_2 & byte_en[3])
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram10_2(clk, entry, write && word_en[2]  && byte_en[2], writedata[87:80], entry, readdata[87:80]);
    ram10_2 = simple_ram(clk, entry, writedata[80:88], write_word_en_2 & byte_en[2])
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram10_1(clk, entry, write && word_en[2]  && byte_en[1], writedata[79:72], entry, readdata[79:72]);
    ram10_1 = simple_ram(clk, entry, writedata[72:80], write_word_en_2 & byte_en[1])
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram10_0(clk, entry, write && word_en[2]  && byte_en[0], writedata[71:64], entry, readdata[71:64]);
    ram10_0 = simple_ram(clk, entry, writedata[64:72], write_word_en_2 & byte_en[0])
    #
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram01_3(clk, entry, write && word_en[1]  && byte_en[3], writedata[63:56], entry, readdata[63:56]);
    write_word_en_1 = write & word_en[1]
    ram01_3 = simple_ram(clk, entry, writedata[56:64], write_word_en_1 & byte_en[3])
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram01_2(clk, entry, write && word_en[1]  && byte_en[2], writedata[55:48], entry, readdata[55:48]);
    ram01_2 = simple_ram(clk, entry, writedata[48:56], write_word_en_1 & byte_en[2])
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram01_1(clk, entry, write && word_en[1]  && byte_en[1], writedata[47:40], entry, readdata[47:40]);
    ram01_1 = simple_ram(clk, entry, writedata[40:48], write_word_en_1 & byte_en[1])
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram01_0(clk, entry, write && word_en[1]  && byte_en[0], writedata[39:32], entry, readdata[39:32]);
    ram01_0 = simple_ram(clk, entry, writedata[32:40], write_word_en_1 & byte_en[0])
    #
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram00_3(clk, entry, write && word_en[0]  && byte_en[3], writedata[31:24], entry, readdata[31:24]);
    write_word_en_0 = write & word_en[0]
    ram00_3 = simple_ram(clk, entry, writedata[24:32], write_word_en_0 & byte_en[3])
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram00_2(clk, entry, write && word_en[0]  && byte_en[2], writedata[23:16], entry, readdata[23:16]);
    ram00_2 = simple_ram(clk, entry, writedata[16:24], write_word_en_0 & byte_en[2])
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram00_1(clk, entry, write && word_en[0]  && byte_en[1], writedata[15: 8], entry, readdata[15:8]);
    ram00_1 = simple_ram(clk, entry, writedata[8:16], write_word_en_0 & byte_en[1])
    #     simple_ram #(.width(8), .widthad(cache_entry)) ram00_0(clk, entry, write && word_en[0]  && byte_en[0], writedata[ 7: 0], entry, readdata[ 7:0]);
    ram00_0 = simple_ram(clk, entry, writedata[0: 8], write_word_en_0 & byte_en[0])

    readdata = Vector.concatRev(ram00_0, ram00_1, ram00_2, ram00_3, ram01_0, ram01_1, ram01_2, ram01_3, ram10_0,
                                ram10_1,
                                ram10_2, ram10_3, ram11_0, ram11_1, ram11_2, ram11_3)
    #
    #
    write_tag_data = If(modify | miss,
                        If(read_miss, Vector.concatRev(VFalse, VTrue, o_tag), Vector.concatRev(VTrue, VTrue, o_tag)),
                        Vector.concatRev(VTrue, VTrue, i_tag))
    #     simple_ram #(.width(25-cache_entry), .widthad(cache_entry)) ram_tag(clk, entry, write, write_tag_data, entry, {dirty, valid, i_tag});
    ram_tag = simple_ram(clk, entry, write_tag_data, write)

    dirty.connect(ram_tag[0])
    valid.connect(ram_tag[1])
    i_tag.connect(ram_tag[2:])

    #
    # `ifdef SIM
    #     integer i;
    #
    #     initial begin
    #         for(i = 0; i <=(2**cache_entry-1); i=i+1) begin
    # 	        ram_tag.mem[i] = 0;
    #         end
    #     end
    # `endif
    #
    # endmodule


    return readdata, wb_addr, hit, modify, miss, valid

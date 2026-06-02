(define (problem picture_106)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery red_pump green_regulator red_pump_1 - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_pump)
        (clear red_battery)
        (clear red_pump_1)
        (part_at red_pump_1 table)
        (part_at red_pump pump_placement)
        (part_at red_battery battery_placement)
        (part_at green_regulator regulator_placement)
        (clear green_regulator)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
        )
    )
)
